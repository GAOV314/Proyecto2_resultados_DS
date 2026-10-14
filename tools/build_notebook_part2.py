"""
Parte 2: agrega modelado con tuning, guardado de artefactos, aplicacion interactiva,
conclusiones y referencias al notebook generado por build_notebook.py (proyecto2-DS_part1.ipynb).
Escribe el resultado final en proyecto2-DS.ipynb

Version "memory-safe": el pre-procesamiento no supervisado que no usa etiquetas (TF-IDF+SVD
para los arboles, y los embeddings de Sentence-BERT) se calcula UNA sola vez sobre todo el
dataset, en vez de refit por cada pliegue/iteracion de busqueda. Esto es valido porque ninguno
de los dos usa la variable objetivo (y), y reduce drasticamente el computo y la memoria pico
en una maquina de 16GB compartida con otros programas.
"""
import nbformat as nbf

nb1 = nbf.read("proyecto2-DS_part1.ipynb", as_version=4)
cells = list(nb1.cells)

def md(src):
    return nbf.v4.new_markdown_cell(src)

def code(src):
    return nbf.v4.new_code_cell(src)

# ======================================================================
# MODELADO
# ======================================================================
cells.append(md("### Modelado — preparación común y estrategia experimental"))
cells.append(md(
"""**Protocolo.** Validación cruzada `StratifiedGroupKFold` agrupada por `essay_id` para el
ajuste de hiperparámetros (3 pliegues, para controlar el tiempo de cómputo) y la comparación
final (5 pliegues); `F1 macro` como métrica principal (objetivo ≥ 0.55); `class_weight='balanced'`
como tratamiento del desbalance; todo el preprocesamiento **que depende de las etiquetas**
(ninguno de los dos, en este proyecto, usa `y`) se ajusta dentro de cada pliegue cuando es viable
en memoria. Además se reserva un *hold-out* del 20 % de los ensayos (`GroupShuffleSplit`) para el
detalle por clase, las matrices de confusión y la aplicación.

**Representaciones.**
- **R1-dispersa (TF-IDF):** `TfidfVectorizer` (unigramas y bigramas, `sublinear_tf`, `min_df=2`) +
  `discourse_type` en *one-hot* + `log1p(len_chars, len_words)`. Se usa para regresión logística y
  SVM lineal, con el vectorizador TF-IDF **ajustado de nuevo dentro de cada pliegue** (es barato en
  memoria por ser disperso).
- **R1-densa (TF-IDF reducido con SVD):** igual que R1, pero reducido a 100 componentes con
  `TruncatedSVD` para los ensambles de árboles (no manejan bien vectores dispersos de alta
  dimensión). Dado que ni el TF-IDF ni el SVD usan la variable objetivo, se calculan **una sola
  vez** sobre todo el corpus (igual que ya se hace, por construcción, con los *embeddings*
  congelados de R2) — esto no introduce fuga de la etiqueta y evita recalcular la misma
  descomposición decenas de veces durante la búsqueda de hiperparámetros, lo cual resultó
  inviable en memoria en una primera corrida de este notebook.
- **R2 (Sentence-BERT):** *embeddings* congelados de `all-MiniLM-L6-v2` (384 dimensiones,
  normalizados) + las mismas variables auxiliares."""
))

cells.append(code(
"""# --- Setup común de modelado (idempotente: re-ejecutable) ---
import pandas as pd, numpy as np, time, warnings, os, json, gc
warnings.filterwarnings('ignore')
from sklearn.model_selection import (StratifiedGroupKFold, GroupShuffleSplit,
                                      cross_validate, RandomizedSearchCV)
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
import matplotlib.pyplot as plt, seaborn as sns
import joblib
sns.set_theme(style='whitegrid')

train = pd.read_csv('data/train.csv')

def limpiar_texto(t):
    t = str(t).strip()
    return (t.replace('Ã\\x82Â´', "'")
             .replace('Ã\\x82Â¨', '"')
             .replace('Ã\\x82Â', "'"))

train['discourse_text_clean'] = train['discourse_text'].apply(limpiar_texto)
train['len_chars'] = train['discourse_text_clean'].str.len()
train['len_words'] = train['discourse_text_clean'].str.split().str.len()
train['log_chars'] = np.log1p(train['len_chars'])
train['log_words'] = np.log1p(train['len_words'])

CLASES = ['Ineffective', 'Adequate', 'Effective']
y = train['discourse_effectiveness']
groups = train['essay_id']

print('train:', train.shape, '| ensayos:', groups.nunique())
print(y.value_counts(normalize=True).round(4).to_dict())

cv_tune = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)   # tuning: mas rapido
cv5 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)       # comparacion final

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, va_idx = next(gss.split(train, y, groups))
print(f'Hold-out agrupado: train={len(tr_idx)} filas / {groups.iloc[tr_idx].nunique()} ensayos, '
      f'test={len(va_idx)} filas / {groups.iloc[va_idx].nunique()} ensayos')
"""
))

cells.append(md(
"""#### Representación R1-dispersa (TF-IDF) — para regresión logística y SVM

`pre_sparse` arma, dentro de cada pliegue, la matriz dispersa de TF-IDF + tipo + longitudes."""
))

cells.append(code(
"""pre_sparse = ColumnTransformer([
    ('tfidf', TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True, min_df=2),
     'discourse_text_clean'),
    ('type', OneHotEncoder(handle_unknown='ignore'), ['discourse_type']),
    ('len', 'passthrough', ['log_chars', 'log_words']),
])
"""
))

cells.append(md(
"""#### Representación R1-densa (TF-IDF + SVD, calculada una sola vez) y R2 (Sentence-BERT)

`R1DenseGlobal` ajusta TF-IDF + SVD (100 componentes) + codificación de tipo **una sola vez sobre
todo el corpus** (ninguno de los dos usa la etiqueta `y`, así que no hay fuga de información de la
variable objetivo). Los *embeddings* de Sentence-BERT (`all-MiniLM-L6-v2`) se cachean en disco
para no recomputar en re-ejecuciones."""
))

cells.append(code(
"""class R1DenseGlobal(BaseEstimator, TransformerMixin):
    \"\"\"TF-IDF -> SVD-100 + one-hot(type) + log-longitudes. Se ajusta UNA vez (no usa y).\"\"\"
    def __init__(self, max_features=15000, n_components=100):
        self.max_features = max_features
        self.n_components = n_components
    def fit(self, X, y=None):
        self.tfidf_ = TfidfVectorizer(max_features=self.max_features, ngram_range=(1, 2),
                                       sublinear_tf=True, min_df=2)
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

r1dense_transformer = R1DenseGlobal(max_features=15000, n_components=100).fit(train)
X_R1_dense = r1dense_transformer.transform(train)
print('X_R1_dense:', X_R1_dense.shape, '(SVD-100 + 7 tipo + 2 longitud)')
gc.collect()
"""
))

cells.append(code(
"""EMB_PATH = 'data/embeddings_minilm6.npy'
if os.path.exists(EMB_PATH):
    E = np.load(EMB_PATH)
    print('Embeddings cargados de cache:', E.shape)
else:
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    E = st_model.encode(train['discourse_text_clean'].tolist(), show_progress_bar=True,
                         normalize_embeddings=True, batch_size=64)
    np.save(EMB_PATH, E)
    print('Embeddings calculados y guardados:', E.shape)
    del st_model
    gc.collect()

ohe_T = OneHotEncoder(handle_unknown='ignore').fit(train[['discourse_type']])
X_R2 = np.hstack([E, ohe_T.transform(train[['discourse_type']]).toarray(),
                   train[['log_chars', 'log_words']].to_numpy()]).astype(np.float32)
print('X_R2:', X_R2.shape, '(384 sBERT + 7 tipo + 2 longitud)')
del E
gc.collect()
"""
))

# ----------------------------------------------------------------
# Hyperparameter tuning
cells.append(md(
"""### Ajuste de hiperparámetros

Se afina cada familia de algoritmos con `RandomizedSearchCV` (`n_jobs=1`, para evitar problemas de
paralelismo por procesos en el mismo entorno donde ya se cargó PyTorch), usando la validación
cruzada agrupada de 3 pliegues (`cv_tune`) y `f1_macro` como criterio."""
))

cells.append(code(
"""rng = 42
tuning_rows = []

# 1) Regresion logistica + R1 (dispersa, pipeline con refit por pliegue)
pipe_logreg_r1 = Pipeline([('pre', pre_sparse),
                            ('clf', LogisticRegression(max_iter=2000, class_weight='balanced'))])
search1 = RandomizedSearchCV(pipe_logreg_r1, {'clf__C': [0.03, 0.1, 0.3, 1, 3, 10]}, n_iter=6,
                              cv=cv_tune, scoring='f1_macro', random_state=rng, n_jobs=1)
search1.fit(train, y, groups=groups)
best_logreg_r1, p1 = search1.best_estimator_, search1.best_params_
tuning_rows.append({'modelo': 'LogReg + R1', 'mejor_f1_macro_cv': search1.best_score_,
                     'mejores_parametros': p1, 'seg_tuning': None})
print(f"[LogReg + R1] mejor F1-macro(cv3)={search1.best_score_:.4f} params={p1}")
del search1; gc.collect()

# 2) SVM lineal + R1 -- se afina el C con LinearSVC "crudo" (rapido, sin calibrar) y SOLO el
#    modelo final se envuelve en CalibratedClassifierCV (para tener probabilidades en la app),
#    asi no se multiplica el costo de la calibracion durante la busqueda.
pipe_svm_r1_raw = Pipeline([('pre', pre_sparse),
                             ('clf', LinearSVC(class_weight='balanced', max_iter=5000))])
search2 = RandomizedSearchCV(pipe_svm_r1_raw, {'clf__C': [0.03, 0.1, 0.3, 1, 3]}, n_iter=5,
                              cv=cv_tune, scoring='f1_macro', random_state=rng, n_jobs=1)
search2.fit(train, y, groups=groups)
p2 = search2.best_params_
tuning_rows.append({'modelo': 'SVM lineal + R1 (búsqueda de C)', 'mejor_f1_macro_cv': search2.best_score_,
                     'mejores_parametros': p2, 'seg_tuning': None})
print(f"[SVM lineal + R1] mejor F1-macro(cv3)={search2.best_score_:.4f} params={p2}")
best_svm_r1 = Pipeline([('pre', pre_sparse),
                         ('clf', CalibratedClassifierCV(
                             LinearSVC(class_weight='balanced', max_iter=5000, C=p2['clf__C']), cv=3))])
del search2, pipe_svm_r1_raw; gc.collect()

# 3) Gradient boosting (HGB) + R1-densa (SVD, ya calculada una sola vez)
search3 = RandomizedSearchCV(HistGradientBoostingClassifier(class_weight='balanced', random_state=rng),
                              {'learning_rate': [0.04, 0.06, 0.1], 'max_leaf_nodes': [31, 63, 127],
                               'max_iter': [120, 200, 300]}, n_iter=6, cv=cv_tune,
                              scoring='f1_macro', random_state=rng, n_jobs=1)
search3.fit(X_R1_dense, y, groups=groups)
best_hgb_r1 = search3.best_estimator_
tuning_rows.append({'modelo': 'HGB + R1densa', 'mejor_f1_macro_cv': search3.best_score_,
                     'mejores_parametros': search3.best_params_, 'seg_tuning': None})
print(f"[HGB + R1densa] mejor F1-macro(cv3)={search3.best_score_:.4f} params={search3.best_params_}")
del search3; gc.collect()

# 4) HGB + R2 (embeddings)
search4 = RandomizedSearchCV(HistGradientBoostingClassifier(class_weight='balanced', random_state=rng),
                              {'learning_rate': [0.04, 0.06, 0.1], 'max_leaf_nodes': [31, 63, 127],
                               'max_iter': [120, 200, 300]}, n_iter=6, cv=cv_tune,
                              scoring='f1_macro', random_state=rng, n_jobs=1)
search4.fit(X_R2, y, groups=groups)
best_hgb_r2 = search4.best_estimator_
tuning_rows.append({'modelo': 'HGB + R2', 'mejor_f1_macro_cv': search4.best_score_,
                     'mejores_parametros': search4.best_params_, 'seg_tuning': None})
print(f"[HGB + R2] mejor F1-macro(cv3)={search4.best_score_:.4f} params={search4.best_params_}")
del search4; gc.collect()

# 5) Regresion logistica + R2 (linea base sobre embeddings)
search5 = RandomizedSearchCV(LogisticRegression(max_iter=2000, class_weight='balanced'),
                              {'C': [0.03, 0.1, 0.3, 1, 3, 10]}, n_iter=6, cv=cv_tune,
                              scoring='f1_macro', random_state=rng, n_jobs=1)
search5.fit(X_R2, y, groups=groups)
best_logreg_r2 = search5.best_estimator_
tuning_rows.append({'modelo': 'LogReg + R2', 'mejor_f1_macro_cv': search5.best_score_,
                     'mejores_parametros': search5.best_params_, 'seg_tuning': None})
print(f"[LogReg + R2] mejor F1-macro(cv3)={search5.best_score_:.4f} params={search5.best_params_}")
del search5; gc.collect()

tuning_df = pd.DataFrame(tuning_rows)
tuning_df
"""
))

cells.append(md(
"""Los mejores hiperparámetros encontrados se reutilizan directamente en la comparación final de
5 pliegues y en el entrenamiento del modelo que se guarda para la aplicación — así el *tuning* no
es un paso aislado, sino parte del mismo *pipeline* que produce el modelo final."""
))

# ----------------------------------------------------------------
cells.append(md("### Comparación de algoritmos — efectividad y tiempos de procesamiento"))
cells.append(code(
"""# --- Modelos sobre R1-dispersa: refit de TF-IDF dentro de cada pliegue (pipeline completo) ---
MODELOS_SPARSE = {
    'Dummy (mayoría)': Pipeline([('pre', pre_sparse), ('clf', DummyClassifier(strategy='most_frequent'))]),
    'LogReg + R1': best_logreg_r1,
    'SVM lineal + R1': best_svm_r1,
}
res = []
for nombre, pipe in MODELOS_SPARSE.items():
    t0 = time.time()
    r = cross_validate(pipe, train, y, groups=groups, cv=cv5,
                        scoring={'f1_macro': 'f1_macro', 'acc': 'accuracy'})
    dt = time.time() - t0
    res.append({'modelo': nombre, 'repr': 'R1 TF-IDF', 'f1_mean': r['test_f1_macro'].mean(),
                'f1_std': r['test_f1_macro'].std(), 'acc_mean': r['test_acc'].mean(),
                'folds': np.round(r['test_f1_macro'], 4).tolist(), 'seg': dt})
    print(f"{nombre}: F1-macro={r['test_f1_macro'].mean():.4f}+/-{r['test_f1_macro'].std():.4f} "
          f"acc={r['test_acc'].mean():.4f} ({dt:.0f}s)")
    gc.collect()
res_sparse = pd.DataFrame(res)

# --- Modelos sobre representaciones densas ya calculadas (R1-densa y R2) ---
MODELOS_DENSE = {
    'HGB + R1densa': (X_R1_dense, best_hgb_r1, 'R1 TF-IDF (denso, SVD)'),
    'RF + R1densa': (X_R1_dense, RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                                          class_weight='balanced_subsample',
                                                          n_jobs=1, random_state=rng), 'R1 TF-IDF (denso, SVD)'),
    'LogReg + R2': (X_R2, best_logreg_r2, 'R2 sBERT'),
    'HGB + R2': (X_R2, best_hgb_r2, 'R2 sBERT'),
}
res2 = []
for nombre, (Xd, clf, repr_tag) in MODELOS_DENSE.items():
    t0 = time.time()
    r = cross_validate(clf, Xd, y, groups=groups, cv=cv5,
                        scoring={'f1_macro': 'f1_macro', 'acc': 'accuracy'})
    dt = time.time() - t0
    res2.append({'modelo': nombre, 'repr': repr_tag, 'f1_mean': r['test_f1_macro'].mean(),
                 'f1_std': r['test_f1_macro'].std(), 'acc_mean': r['test_acc'].mean(),
                 'folds': np.round(r['test_f1_macro'], 4).tolist(), 'seg': dt})
    print(f"{nombre}: F1-macro={r['test_f1_macro'].mean():.4f}+/-{r['test_f1_macro'].std():.4f} "
          f"acc={r['test_acc'].mean():.4f} ({dt:.0f}s)")
    gc.collect()
res_dense = pd.DataFrame(res2)

compar = pd.concat([res_sparse, res_dense], ignore_index=True).sort_values('f1_mean', ascending=False).reset_index(drop=True)
compar[['modelo', 'repr', 'f1_mean', 'f1_std', 'acc_mean', 'seg']].round(4)
"""
))

cells.append(code(
"""# --- Grafico 1: F1-macro con objetivo 0.55 ---
fig, ax = plt.subplots(figsize=(10, 4.5))
o = compar.sort_values('f1_mean')
ax.barh(o['modelo'], o['f1_mean'], xerr=o['f1_std'], capsize=4, color=sns.color_palette('viridis', len(o)))
ax.axvline(0.55, color='#B23A48', ls='--', label='Objetivo F1-macro 0.55')
for i, (v, s) in enumerate(zip(o['f1_mean'], o['f1_std'])):
    ax.text(v + s + 0.005, i, f"{v:.3f}+/-{s:.3f}", va='center', fontsize=9)
ax.set_xlabel('F1-macro (media CV 5 pliegues agrupados por ensayo)')
ax.set_title('Comparación de modelos — F1-macro (tras ajuste de hiperparámetros)')
ax.legend(); fig.tight_layout()
fig.savefig('reporte_fig_f1_macro.png', dpi=130, bbox_inches='tight')
plt.show()

ganador = compar.iloc[0]['modelo']
print(f"Ganador por F1-macro: {ganador} ({compar.iloc[0]['f1_mean']:.4f}+/-{compar.iloc[0]['f1_std']:.4f}, "
      f"repr={compar.iloc[0]['repr']})")
"""
))

cells.append(code(
"""# --- Grafico 2: tiempos de procesamiento (ajuste + CV 5 pliegues) ---
fig, ax = plt.subplots(figsize=(9, 4.2))
o2 = compar.sort_values('seg')
ax.barh(o2['modelo'], o2['seg'], color=sns.color_palette('mako', len(o2)))
for i, v in enumerate(o2['seg']):
    ax.text(v + max(o2['seg']) * 0.01, i, f"{v:.0f}s", va='center', fontsize=9)
ax.set_xlabel('Segundos (5 ajustes + predicción, validación cruzada agrupada)')
ax.set_title('Tiempos de procesamiento por modelo')
fig.tight_layout()
fig.savefig('reporte_fig_tiempos.png', dpi=130, bbox_inches='tight')
plt.show()
"""
))

cells.append(md("#### Estabilidad entre pliegues y desempeño por clase (conjunto de prueba agrupado)"))
cells.append(code(
"""# --- Estabilidad por pliegue ---
fig, ax = plt.subplots(figsize=(10, 3.8))
rows = []
for _, r in compar.iterrows():
    for f, v in enumerate(r['folds']):
        rows.append({'modelo': r['modelo'], 'fold': f + 1, 'f1': v})
long = pd.DataFrame(rows)
sns.stripplot(data=long, x='f1', y='modelo', hue='modelo', size=8, alpha=0.8, legend=False, ax=ax)
ax.axvline(0.55, color='#B23A48', ls='--')
ax.set_title('Estabilidad: F1-macro por pliegue (CV agrupada por ensayo)')
fig.tight_layout()
fig.savefig('reporte_fig_estabilidad.png', dpi=130, bbox_inches='tight')
plt.show()

# --- Entrenamiento final sobre el hold-out 80/20 agrupado para todos los modelos del nucleo ---
FUENTE_SPARSE = set(MODELOS_SPARSE.keys())
holdout_rows = []
modelos_entrenados = {}
for nombre in list(MODELOS_SPARSE.keys()) + list(MODELOS_DENSE.keys()):
    if nombre in MODELOS_DENSE:
        Xd, clf, _ = MODELOS_DENSE[nombre]
        clf.fit(Xd[tr_idx], y.iloc[tr_idx])
        pred = clf.predict(Xd[va_idx])
        tag = 'R1dense' if 'R1densa' in nombre else 'R2'
        modelos_entrenados[nombre] = (tag, clf)
    else:
        pipe = MODELOS_SPARSE[nombre]
        pipe.fit(train.iloc[tr_idx], y.iloc[tr_idx])
        pred = pipe.predict(train.iloc[va_idx])
        modelos_entrenados[nombre] = ('R1sparse', pipe)
    for did, eid, real, p in zip(train['discourse_id'].iloc[va_idx], train['essay_id'].iloc[va_idx],
                                   y.iloc[va_idx], pred):
        holdout_rows.append({'modelo': nombre, 'discourse_id': did, 'essay_id': eid, 'real': real, 'pred': p})
    acc = accuracy_score(y.iloc[va_idx], pred); f1m = f1_score(y.iloc[va_idx], pred, average='macro')
    print(f"\\n== {nombre} (hold-out 20% ensayos) == acc={acc:.4f} f1_macro={f1m:.4f}")
    print(classification_report(y.iloc[va_idx], pred, digits=3))
    gc.collect()

holdout_df = pd.DataFrame(holdout_rows)
os.makedirs('artifacts', exist_ok=True)
holdout_df.to_csv('artifacts/holdout_predictions.csv', index=False)
print('\\nPredicciones de hold-out guardadas en artifacts/holdout_predictions.csv:', holdout_df.shape)
"""
))

cells.append(code(
"""# --- Matrices de confusion (top-3 por F1-macro, hold-out agrupado) ---
TOP3 = compar.head(3)['modelo'].tolist()
fig, axes = plt.subplots(1, len(TOP3), figsize=(5.2 * len(TOP3), 4.2))
if len(TOP3) == 1:
    axes = [axes]
for ax, nombre in zip(axes, TOP3):
    sub = holdout_df[holdout_df['modelo'] == nombre]
    cm = confusion_matrix(sub['real'], sub['pred'], labels=CLASES, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=CLASES, yticklabels=CLASES, cmap='Blues', ax=ax)
    acc_ = accuracy_score(sub['real'], sub['pred']); f1_ = f1_score(sub['real'], sub['pred'], average='macro')
    ax.set_title(f"{nombre}\\nacc={acc_:.3f} f1m={f1_:.3f}")
    ax.set_xlabel('Predicho'); ax.set_ylabel('Real')
fig.suptitle('Matrices de confusión normalizadas por fila (hold-out agrupado por ensayo)', y=1.03)
fig.tight_layout()
fig.savefig('reporte_fig_matrices_confusion.png', dpi=130, bbox_inches='tight')
plt.show()
"""
))

cells.append(code(
"""from IPython.display import Markdown, display

ganador_row = compar.iloc[0]
segundo_row = compar.iloc[1]
texto = f\"\"\"### Comparación, modelo seleccionado y razones

> Números verificados con CV de 5 pliegues agrupada por `essay_id`, tras el ajuste de
> hiperparámetros reportado arriba (esta ejecución).

**Ranking por F1-macro:**

{chr(10).join(f"{i+1}. **{r.modelo}** ({r.repr}) — F1-macro={r.f1_mean:.4f}+/-{r.f1_std:.4f}, "
              f"acc={r.acc_mean:.4f}, tiempo={r.seg:.0f}s" for i, r in compar.iterrows())}

**Modelo seleccionado: `{ganador_row['modelo']}`.** Razones: (1) el mayor F1-macro
({ganador_row['f1_mean']:.4f}+/-{ganador_row['f1_std']:.4f}), superando el objetivo de 0.55 con
margen (si el ganador lo alcanza; de lo contrario, se discute como limitación en Conclusiones);
(2) estabilidad razonable entre pliegues (ver gráfico de estabilidad); (3) coherente con la
revisión bibliográfica del grupo, que anticipaba que los ensambles de árboles o las
representaciones semánticas capturarían señal adicional frente a una línea base puramente lineal.
El segundo lugar, `{segundo_row['modelo']}`, queda como alternativa de respaldo — útil si se
necesitara mayor interpretabilidad o menor tiempo de entrenamiento.

**Sobre la métrica de accuracy:** nótese que un modelo puede tener *accuracy* relativamente alta
sesgando sus predicciones hacia la clase mayoritaria (`Adequate`) mientras su F1-macro es bajo —
exactamente el motivo por el que el objetivo del proyecto se definió en términos de F1-macro y no
de accuracy (He y Garcia, 2009).
\"\"\"
display(Markdown(texto))
"""
))

nbf.write(nbf.v4.new_notebook(cells=cells, metadata=nb1.metadata), "proyecto2-DS_part2.ipynb")
print("part2 cells:", len(cells))
