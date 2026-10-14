"""Smoke test de la version 'memory-safe' (R1DenseGlobal + busquedas inline), con muestra chica."""
import pandas as pd, numpy as np, time, warnings, os, gc
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
from sklearn.metrics import f1_score, accuracy_score, classification_report

train_full = pd.read_csv('data/train.csv')

def limpiar_texto(t):
    t = str(t).strip()
    return (t.replace('Ã\x82Â´', "'").replace('Ã\x82Â¨', '"').replace('Ã\x82Â', "'"))

train_full['discourse_text_clean'] = train_full['discourse_text'].apply(limpiar_texto)
essays = train_full['essay_id'].drop_duplicates().sample(500, random_state=0)
train = train_full[train_full['essay_id'].isin(essays)].reset_index(drop=True)
train['len_chars'] = train['discourse_text_clean'].str.len()
train['len_words'] = train['discourse_text_clean'].str.split().str.len()
train['log_chars'] = np.log1p(train['len_chars'])
train['log_words'] = np.log1p(train['len_words'])
print('sample:', train.shape, 'ensayos:', train['essay_id'].nunique())

CLASES = ['Ineffective', 'Adequate', 'Effective']
y = train['discourse_effectiveness']
groups = train['essay_id']
cv_tune = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
cv5 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, va_idx = next(gss.split(train, y, groups))

pre_sparse = ColumnTransformer([
    ('tfidf', TfidfVectorizer(max_features=1500, ngram_range=(1, 2), sublinear_tf=True, min_df=2), 'discourse_text_clean'),
    ('type', OneHotEncoder(handle_unknown='ignore'), ['discourse_type']),
    ('len', 'passthrough', ['log_chars', 'log_words']),
])

class R1DenseGlobal(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=1500, n_components=16):
        self.max_features = max_features; self.n_components = n_components
    def fit(self, X, y=None):
        self.tfidf_ = TfidfVectorizer(max_features=self.max_features, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
        self.svd_ = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.svd_.fit(self.tfidf_.fit_transform(X['discourse_text_clean']))
        self.ohe_ = OneHotEncoder(handle_unknown='ignore').fit(X[['discourse_type']])
        return self
    def transform(self, X):
        return np.hstack([
            self.svd_.transform(self.tfidf_.transform(X['discourse_text_clean'])),
            self.ohe_.transform(X[['discourse_type']]).toarray(),
            X[['log_chars', 'log_words']].to_numpy(),
        ]).astype(np.float32)

r1dense_transformer = R1DenseGlobal(max_features=1500, n_components=16).fit(train)
X_R1_dense = r1dense_transformer.transform(train)
print('X_R1_dense', X_R1_dense.shape)

rng_ = np.random.RandomState(0)
E = rng_.normal(size=(len(train), 32)); E = E / np.linalg.norm(E, axis=1, keepdims=True)
ohe_T = OneHotEncoder(handle_unknown='ignore').fit(train[['discourse_type']])
X_R2 = np.hstack([E, ohe_T.transform(train[['discourse_type']]).toarray(), train[['log_chars', 'log_words']].to_numpy()]).astype(np.float32)

rng = 42
tuning_rows = []

pipe_logreg_r1 = Pipeline([('pre', pre_sparse), ('clf', LogisticRegression(max_iter=500, class_weight='balanced'))])
search1 = RandomizedSearchCV(pipe_logreg_r1, {'clf__C': [0.1, 1]}, n_iter=2, cv=cv_tune, scoring='f1_macro', random_state=rng, n_jobs=1)
search1.fit(train, y, groups=groups)
best_logreg_r1, p1 = search1.best_estimator_, search1.best_params_
print('LogReg R1 OK', p1)

pipe_svm_r1_raw = Pipeline([('pre', pre_sparse), ('clf', LinearSVC(class_weight='balanced', max_iter=2000))])
search2 = RandomizedSearchCV(pipe_svm_r1_raw, {'clf__C': [0.1, 1]}, n_iter=2, cv=cv_tune, scoring='f1_macro', random_state=rng, n_jobs=1)
search2.fit(train, y, groups=groups)
p2 = search2.best_params_
best_svm_r1 = Pipeline([('pre', pre_sparse), ('clf', CalibratedClassifierCV(LinearSVC(class_weight='balanced', max_iter=2000, C=p2['clf__C']), cv=2))])
print('SVM R1 OK', p2)

search3 = RandomizedSearchCV(HistGradientBoostingClassifier(class_weight='balanced', random_state=rng),
                              {'learning_rate': [0.06, 0.1], 'max_leaf_nodes': [15, 31], 'max_iter': [20, 40]},
                              n_iter=2, cv=cv_tune, scoring='f1_macro', random_state=rng, n_jobs=1)
search3.fit(X_R1_dense, y, groups=groups)
best_hgb_r1 = search3.best_estimator_
print('HGB R1dense OK', search3.best_params_)

search4 = RandomizedSearchCV(HistGradientBoostingClassifier(class_weight='balanced', random_state=rng),
                              {'learning_rate': [0.06, 0.1], 'max_leaf_nodes': [15, 31], 'max_iter': [20, 40]},
                              n_iter=2, cv=cv_tune, scoring='f1_macro', random_state=rng, n_jobs=1)
search4.fit(X_R2, y, groups=groups)
best_hgb_r2 = search4.best_estimator_
print('HGB R2 OK', search4.best_params_)

search5 = RandomizedSearchCV(LogisticRegression(max_iter=500, class_weight='balanced'), {'C': [0.1, 1]},
                              n_iter=2, cv=cv_tune, scoring='f1_macro', random_state=rng, n_jobs=1)
search5.fit(X_R2, y, groups=groups)
best_logreg_r2 = search5.best_estimator_
print('LogReg R2 OK', search5.best_params_)

MODELOS_SPARSE = {
    'Dummy (mayoría)': Pipeline([('pre', pre_sparse), ('clf', DummyClassifier(strategy='most_frequent'))]),
    'LogReg + R1': best_logreg_r1,
    'SVM lineal + R1': best_svm_r1,
}
res = []
for nombre, pipe in MODELOS_SPARSE.items():
    r = cross_validate(pipe, train, y, groups=groups, cv=cv5, scoring={'f1_macro': 'f1_macro', 'acc': 'accuracy'})
    res.append({'modelo': nombre, 'repr': 'R1 TF-IDF', 'f1_mean': r['test_f1_macro'].mean(), 'f1_std': r['test_f1_macro'].std(),
                'acc_mean': r['test_acc'].mean(), 'folds': np.round(r['test_f1_macro'], 4).tolist(), 'seg': 1.0})
    print(nombre, 'cv OK')
res_sparse = pd.DataFrame(res)

MODELOS_DENSE = {
    'HGB + R1densa': (X_R1_dense, best_hgb_r1, 'R1 TF-IDF (denso, SVD)'),
    'RF + R1densa': (X_R1_dense, RandomForestClassifier(n_estimators=20, min_samples_leaf=2, class_weight='balanced_subsample', n_jobs=1, random_state=rng), 'R1 TF-IDF (denso, SVD)'),
    'LogReg + R2': (X_R2, best_logreg_r2, 'R2 sBERT'),
    'HGB + R2': (X_R2, best_hgb_r2, 'R2 sBERT'),
}
res2 = []
for nombre, (Xd, clf, repr_tag) in MODELOS_DENSE.items():
    r = cross_validate(clf, Xd, y, groups=groups, cv=cv5, scoring={'f1_macro': 'f1_macro', 'acc': 'accuracy'})
    res2.append({'modelo': nombre, 'repr': repr_tag, 'f1_mean': r['test_f1_macro'].mean(), 'f1_std': r['test_f1_macro'].std(),
                 'acc_mean': r['test_acc'].mean(), 'folds': np.round(r['test_f1_macro'], 4).tolist(), 'seg': 1.0})
    print(nombre, 'cv OK')
res_dense = pd.DataFrame(res2)

compar = pd.concat([res_sparse, res_dense], ignore_index=True).sort_values('f1_mean', ascending=False).reset_index(drop=True)
print(compar[['modelo', 'repr', 'f1_mean', 'acc_mean']])

holdout_rows = []
modelos_entrenados = {}
for nombre in list(MODELOS_SPARSE.keys()) + list(MODELOS_DENSE.keys()):
    if nombre in MODELOS_DENSE:
        Xd, clf, _ = MODELOS_DENSE[nombre]
        clf.fit(Xd[tr_idx], y.iloc[tr_idx]); pred = clf.predict(Xd[va_idx])
        tag = 'R1dense' if 'R1densa' in nombre else 'R2'
        modelos_entrenados[nombre] = (tag, clf)
    else:
        pipe = MODELOS_SPARSE[nombre]
        pipe.fit(train.iloc[tr_idx], y.iloc[tr_idx]); pred = pipe.predict(train.iloc[va_idx])
        modelos_entrenados[nombre] = ('R1sparse', pipe)
    acc = accuracy_score(y.iloc[va_idx], pred); f1m = f1_score(y.iloc[va_idx], pred, average='macro')
    print(nombre, 'holdout acc', round(acc,3), 'f1m', round(f1m,3), 'proba:', hasattr(modelos_entrenados[nombre][1], 'predict_proba'))

# Simular la clasificacion de un texto nuevo con cada tag, como en la app
fila = pd.DataFrame([{'discourse_text_clean': 'this is a test argument about school policy',
                       'discourse_type': 'Claim', 'log_chars': 3.0, 'log_words': 2.0}])
for nombre, (tag, modelo) in modelos_entrenados.items():
    if tag == 'R1sparse':
        p = modelo.predict(fila)[0]
    elif tag == 'R1dense':
        x = r1dense_transformer.transform(fila)
        p = modelo.predict(x)[0]
    else:
        x = np.hstack([np.zeros((1,32)), ohe_T.transform(fila[['discourse_type']]).toarray(), fila[['log_chars','log_words']].to_numpy()])
        p = modelo.predict(x)[0]
    print('clasificar nuevo texto ->', nombre, ':', p)

print('\\nSMOKE TEST 2 PASSED')
