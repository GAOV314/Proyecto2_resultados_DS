"""Smoke test rapido (muestra pequena, pocos folds) para validar que la logica de modelado
(ColumnTransformer, R1Dense, CalibratedClassifierCV, RandomizedSearchCV con groups, etc.)
corre sin errores antes de lanzar la ejecucion completa del notebook."""
import pandas as pd, numpy as np, time, warnings, os
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit, cross_validate, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

train_full = pd.read_csv('data/train.csv')

def limpiar_texto(t):
    t = str(t).strip()
    return (t.replace('Ã\x82Â´', "'").replace('Ã\x82Â¨', '"').replace('Ã\x82Â', "'"))

train_full['discourse_text_clean'] = train_full['discourse_text'].apply(limpiar_texto)

# muestra pequena pero conservando ensayos completos (agrupado)
essays = train_full['essay_id'].drop_duplicates().sample(400, random_state=0)
train = train_full[train_full['essay_id'].isin(essays)].reset_index(drop=True)
train['len_chars'] = train['discourse_text_clean'].str.len()
train['len_words'] = train['discourse_text_clean'].str.split().str.len()
train['log_chars'] = np.log1p(train['len_chars'])
train['log_words'] = np.log1p(train['len_words'])
print('smoke sample:', train.shape, 'ensayos:', train['essay_id'].nunique())

CLASES = ['Ineffective', 'Adequate', 'Effective']
y = train['discourse_effectiveness']
groups = train['essay_id']
cv_tune = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
cv5 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, va_idx = next(gss.split(train, y, groups))

pre_sparse = ColumnTransformer([
    ('tfidf', TfidfVectorizer(max_features=2000, ngram_range=(1, 2), sublinear_tf=True, min_df=2), 'discourse_text_clean'),
    ('type', OneHotEncoder(handle_unknown='ignore'), ['discourse_type']),
    ('len', 'passthrough', ['log_chars', 'log_words']),
])

class R1Dense(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.tfidf_ = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
        self.svd_ = TruncatedSVD(n_components=16, random_state=42)
        self.svd_.fit(self.tfidf_.fit_transform(X['discourse_text_clean']))
        self.ohe_ = OneHotEncoder(handle_unknown='ignore').fit(X[['discourse_type']])
        return self
    def transform(self, X):
        return np.hstack([
            self.svd_.transform(self.tfidf_.transform(X['discourse_text_clean'])),
            self.ohe_.transform(X[['discourse_type']]).toarray(),
            X[['log_chars', 'log_words']].to_numpy(),
        ])

# R2 proxy rapido (sin descargar nada): vector aleatorio reproducible del mismo tamano que sBERT
rng_ = np.random.RandomState(0)
E = rng_.normal(size=(len(train), 32))
E = E / np.linalg.norm(E, axis=1, keepdims=True)
ohe_T = OneHotEncoder(handle_unknown='ignore').fit(train[['discourse_type']])
X_R2 = np.hstack([E, ohe_T.transform(train[['discourse_type']]).toarray(), train[['log_chars', 'log_words']].to_numpy()])

rng = 42
tuning_rows = []
def tune(nombre, estimator, X, yv, grupos_v, param_dist, n_iter=3):
    t0 = time.time()
    search = RandomizedSearchCV(estimator, param_dist, n_iter=n_iter, cv=cv_tune, scoring='f1_macro',
                                 random_state=rng, n_jobs=-1, refit=True)
    search.fit(X, yv, groups=grupos_v)
    dt = time.time() - t0
    tuning_rows.append({'modelo': nombre, 'mejor_f1_macro_cv': search.best_score_, 'mejores_parametros': search.best_params_, 'seg_tuning': dt})
    print(f"[{nombre}] mejor F1-macro={search.best_score_:.4f} params={search.best_params_} ({dt:.1f}s)")
    return search.best_estimator_, search.best_params_

pipe_logreg_r1 = Pipeline([('pre', pre_sparse), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))])
best_logreg_r1, p1 = tune('LogReg + R1', pipe_logreg_r1, train, y, groups, {'clf__C': [0.1, 1, 3]}, n_iter=2)

pipe_svm_r1_raw = Pipeline([('pre', pre_sparse), ('clf', LinearSVC(class_weight='balanced', max_iter=3000))])
_, p2 = tune('SVM (busqueda C)', pipe_svm_r1_raw, train, y, groups, {'clf__C': [0.1, 1]}, n_iter=2)
best_svm_r1 = Pipeline([('pre', pre_sparse), ('clf', CalibratedClassifierCV(LinearSVC(class_weight='balanced', max_iter=3000, C=p2['clf__C']), cv=2))])

pipe_hgb_r1 = Pipeline([('feat', R1Dense()), ('clf', HistGradientBoostingClassifier(class_weight='balanced', random_state=rng))])
best_hgb_r1, p3 = tune('HGB + R1densa', pipe_hgb_r1, train, y, groups,
                        {'clf__learning_rate': [0.06, 0.1], 'clf__max_leaf_nodes': [15, 31], 'clf__max_iter': [20, 40]}, n_iter=2)

best_hgb_r2, p4 = tune('HGB + R2', HistGradientBoostingClassifier(class_weight='balanced', random_state=rng), X_R2, y, groups,
                        {'learning_rate': [0.06, 0.1], 'max_leaf_nodes': [15, 31], 'max_iter': [20, 40]}, n_iter=2)

best_logreg_r2, p5 = tune('LogReg + R2', LogisticRegression(max_iter=1000, class_weight='balanced'), X_R2, y, groups,
                           {'C': [0.1, 1, 3]}, n_iter=2)

tuning_df = pd.DataFrame(tuning_rows)
print(tuning_df)

MODELOS_R1 = {
    'Dummy (mayoría)': Pipeline([('pre', pre_sparse), ('clf', DummyClassifier(strategy='most_frequent'))]),
    'LogReg + R1': best_logreg_r1,
    'SVM lineal + R1': best_svm_r1,
    'HGB + R1densa': best_hgb_r1,
    'RF + R1densa': Pipeline([('feat', R1Dense()), ('clf', RandomForestClassifier(n_estimators=30, min_samples_leaf=2, class_weight='balanced_subsample', n_jobs=-1, random_state=rng))]),
}
res = []
for nombre, pipe in MODELOS_R1.items():
    t0 = time.time()
    r = cross_validate(pipe, train, y, groups=groups, cv=cv5, scoring={'f1_macro': 'f1_macro', 'acc': 'accuracy'})
    dt = time.time() - t0
    res.append({'modelo': nombre, 'repr': 'R1 TF-IDF', 'f1_mean': r['test_f1_macro'].mean(), 'f1_std': r['test_f1_macro'].std(), 'acc_mean': r['test_acc'].mean(), 'folds': np.round(r['test_f1_macro'], 4).tolist(), 'seg': dt})
    print(nombre, 'OK', dt, 's')
res_R1 = pd.DataFrame(res)

MODELOS_R2 = {'LogReg + R2': best_logreg_r2, 'HGB + R2': best_hgb_r2}
res2 = []
for nombre, clf in MODELOS_R2.items():
    t0 = time.time()
    r = cross_validate(clf, X_R2, y, groups=groups, cv=cv5, scoring={'f1_macro': 'f1_macro', 'acc': 'accuracy'})
    dt = time.time() - t0
    res2.append({'modelo': nombre, 'repr': 'R2 sBERT', 'f1_mean': r['test_f1_macro'].mean(), 'f1_std': r['test_f1_macro'].std(), 'acc_mean': r['test_acc'].mean(), 'folds': np.round(r['test_f1_macro'], 4).tolist(), 'seg': dt})
    print(nombre, 'OK', dt, 's')
res_R2 = pd.DataFrame(res2)

compar = pd.concat([res_R1, res_R2], ignore_index=True).sort_values('f1_mean', ascending=False).reset_index(drop=True)
print(compar[['modelo', 'repr', 'f1_mean', 'acc_mean', 'seg']])

# entrenamiento final + holdout (como en el notebook real)
TODOS = list(MODELOS_R1.keys()) + list(MODELOS_R2.keys())
holdout_rows = []
modelos_entrenados = {}
for nombre in TODOS:
    if nombre in MODELOS_R2:
        clf = MODELOS_R2[nombre]
        clf.fit(X_R2[tr_idx], y.iloc[tr_idx]); pred = clf.predict(X_R2[va_idx])
        modelos_entrenados[nombre] = ('R2', clf)
    else:
        pipe = MODELOS_R1[nombre]
        pipe.fit(train.iloc[tr_idx], y.iloc[tr_idx]); pred = pipe.predict(train.iloc[va_idx])
        modelos_entrenados[nombre] = ('R1', pipe)
    acc = accuracy_score(y.iloc[va_idx], pred); f1m = f1_score(y.iloc[va_idx], pred, average='macro')
    print(nombre, 'holdout acc', acc, 'f1m', f1m)
    # ¿predict_proba disponible?
    has_proba = hasattr(modelos_entrenados[nombre][1], 'predict_proba')
    print('  predict_proba:', has_proba)

print('\\nSMOKE TEST PASSED')
