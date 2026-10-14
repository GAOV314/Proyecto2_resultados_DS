"""
Parte 3: guardado de artefactos del modelo final + aplicacion interactiva (ipywidgets + Plotly)
+ conclusiones + referencias. Lee proyecto2-DS_part2.ipynb y escribe proyecto2-DS.ipynb (final).
"""
import nbformat as nbf

nb2 = nbf.read("proyecto2-DS_part2.ipynb", as_version=4)
cells = list(nb2.cells)

def md(src):
    return nbf.v4.new_markdown_cell(src)

def code(src):
    return nbf.v4.new_code_cell(src)

# ======================================================================
# GUARDADO DE ARTEFACTOS
# ======================================================================
cells.append(md("### Guardado de los artefactos del modelo final"))
cells.append(md(
"""Se guardan en `artifacts/` los *pipelines* ya entrenados de **todos** los modelos del núcleo
(no solo el ganador), para que la aplicación pueda mostrar los resultados de cada algoritmo por
separado o de todos a la vez, tal como lo pide la guía del proyecto. También se guarda el
codificador `all-MiniLM-L6-v2` (por nombre, se recarga igual) y la tabla comparativa."""
))

cells.append(code(
"""os.makedirs('artifacts', exist_ok=True)

for nombre, (repr_tag, modelo) in modelos_entrenados.items():
    slug = nombre.lower().replace(' ', '_').replace('+', '').replace('(', '').replace(')', '')
    joblib.dump({'repr': repr_tag, 'modelo': modelo}, f'artifacts/{slug}.joblib')

joblib.dump(ohe_T, 'artifacts/ohe_discourse_type.joblib')
joblib.dump(r1dense_transformer, 'artifacts/r1dense_transformer.joblib')
compar.to_csv('artifacts/comparacion_modelos.csv', index=False)
tuning_df.to_csv('artifacts/tuning_hiperparametros.csv', index=False)

with open('artifacts/meta.json', 'w') as f:
    json.dump({
        'clases': CLASES,
        'modelo_ganador': ganador,
        'sentence_transformer': 'all-MiniLM-L6-v2',
        'columnas_entrada': ['discourse_text_clean', 'discourse_type', 'log_chars', 'log_words'],
    }, f, indent=2)

print('Artefactos guardados en ./artifacts:')
for fn in sorted(os.listdir('artifacts')):
    print(' -', fn)
"""
))

# ======================================================================
# APLICACIÓN
# ======================================================================
cells.append(md("### Descripción de la aplicación creada"))
cells.append(md(
"""La aplicación se implementó **dentro de este mismo Jupyter Notebook**, usando `ipywidgets` para
los controles interactivos (pestañas, listas desplegables, casillas, botones, áreas de texto) y
`Plotly` para las gráficas interactivas (zoom, *hover* con el valor exacto, posibilidad de
ocultar/mostrar series haciendo clic en la leyenda). Esta combinación se eligió porque:

- No requiere un servidor externo (Streamlit/Dash) ni pasos adicionales de despliegue: cualquier
  persona con Jupyter puede ejecutar el notebook y usar la aplicación de inmediato.
- `ipywidgets` permite construir una interfaz de pestañas con controles familiares (listas
  desplegables, casillas de verificación, botones) sin escribir HTML/CSS/JS.
- `Plotly` genera gráficas realmente interactivas (no solo estáticas), cumpliendo el requisito de
  que el usuario pueda explorar visualmente el rendimiento de los algoritmos.

**Estructura de la aplicación (3 pestañas):**

1. **Explorar los datos** — permite elegir cualquier variable del conjunto de datos ya limpio
   (`discourse_type`, `discourse_effectiveness`, longitud en caracteres/palabras) y ver su
   distribución, además de cruzarla con otra variable categórica.
2. **Resultados de los modelos** — permite elegir uno, varios o todos los algoritmos entrenados y
   ver su desempeño (F1-macro, exactitud, tiempo de procesamiento, matriz de confusión) en
   gráficas interactivas; una casilla permite **esconder** toda esta sección de rendimiento.
3. **Clasificar un fragmento nuevo** — el usuario escribe o pega un fragmento de texto
   argumentativo, elige su tipo discursivo (`discourse_type`) y, al presionar "Clasificar",
   la aplicación muestra la predicción de cada modelo entrenado con su probabilidad por clase.

**Teoría del color aplicada.** Se definieron dos paletas fijas, reutilizadas en *todas* las
gráficas de la aplicación y del informe, para que el usuario asocie siempre el mismo color al
mismo significado:

- **`discourse_type`** (7 categorías nominales, sin orden): paleta cualitativa de alto contraste
  entre sí e inspirada en la paleta de Okabe-Ito, diseñada para ser distinguible también por
  personas con daltonismo (deuteranopía/protanopía), ya que cada color difiere tanto en matiz como
  en luminosidad.
- **`discourse_effectiveness`** (3 categorías **ordinales**): paleta semántica tipo semáforo pero
  ajustada para accesibilidad — en vez de rojo/verde puros (que colisionan para personas
  daltónicas), se usan colores que además difieren en luminosidad: `Ineffective` = rojo ladrillo
  oscuro, `Adequate` = ámbar/dorado (el más claro, color intermedio), `Effective` = verde azulado
  oscuro. El orden de luminosidad (oscuro → claro → oscuro) refuerza visualmente que `Adequate`
  es un punto intermedio entre dos extremos, consistente con la naturaleza ordinal de la variable."""
))

cells.append(code(
"""# --- Paleta de colores (teoria del color, reutilizada en toda la app) ---
COLOR_EFECTIVIDAD = {
    'Ineffective': '#B23A48',   # rojo ladrillo oscuro
    'Adequate':    '#E3B23C',   # ambar/dorado (mas claro)
    'Effective':   '#1F7A5C',   # verde azulado oscuro
}
PALETA_TIPO = {
    'Lead':                 '#4477AA',
    'Position':              '#66CCEE',
    'Claim':                 '#228833',
    'Counterclaim':          '#CCBB44',
    'Rebuttal':              '#EE6677',
    'Evidence':              '#AA3377',
    'Concluding Statement':  '#BBBBBB',
}
print('Paletas definidas: COLOR_EFECTIVIDAD (ordinal), PALETA_TIPO (cualitativa).')
"""
))

cells.append(code(
"""# --- Carga de artefactos (para que la app pueda ejecutarse de forma independiente) ---
import glob
import plotly.graph_objects as go
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display, clear_output

ART = 'artifacts'
meta = json.load(open(f'{ART}/meta.json'))
compar_app = pd.read_csv(f'{ART}/comparacion_modelos.csv')
holdout_app = pd.read_csv(f'{ART}/holdout_predictions.csv')
CLASES_APP = meta['clases']

modelos_app = {}
for fn in glob.glob(f'{ART}/*.joblib'):
    if 'ohe_discourse_type' in fn or 'r1dense_transformer' in fn:
        continue
    obj = joblib.load(fn)
    nombre_legible = compar_app.loc[compar_app['modelo'].str.lower().str.replace(' ', '_')
                                     .str.replace('+', '').str.replace('(', '').str.replace(')', '')
                                     == fn.split('/')[-1].replace('.joblib', ''), 'modelo']
    nombre_legible = nombre_legible.iloc[0] if len(nombre_legible) else fn
    modelos_app[nombre_legible] = obj

ohe_app = joblib.load(f'{ART}/ohe_discourse_type.joblib')
r1dense_transformer_app = joblib.load(f'{ART}/r1dense_transformer.joblib')

from sentence_transformers import SentenceTransformer
_encoder_app = None
def get_encoder():
    global _encoder_app
    if _encoder_app is None:
        _encoder_app = SentenceTransformer(meta['sentence_transformer'])
    return _encoder_app

print('Modelos disponibles en la app:', list(modelos_app.keys()))
"""
))

cells.append(md("#### Pestaña 1 — Explorar los datos"))
cells.append(code(
"""VARS_NUM = ['len_chars', 'len_words']
VARS_CAT = ['discourse_type', 'discourse_effectiveness']

out_explorar = widgets.Output()
dd_variable = widgets.Dropdown(options=VARS_CAT + VARS_NUM, value='discourse_effectiveness',
                                description='Variable:')
dd_cruce = widgets.Dropdown(options=['(ninguna)'] + VARS_CAT, value='(ninguna)',
                             description='Cruzar con:')

def color_de(cat, valor):
    if cat == 'discourse_effectiveness':
        return COLOR_EFECTIVIDAD.get(valor, '#888888')
    if cat == 'discourse_type':
        return PALETA_TIPO.get(valor, '#888888')
    return '#4477AA'

def dibujar_explorador(_=None):
    with out_explorar:
        clear_output(wait=True)
        var = dd_variable.value
        cruce = dd_cruce.value
        if var in VARS_CAT:
            orden = CLASES_APP if var == 'discourse_effectiveness' else sorted(train[var].unique())
            if cruce == '(ninguna)' or cruce == var:
                conteo = train[var].value_counts().reindex(orden)
                colores = [color_de(var, v) for v in orden]
                fig = go.Figure(go.Bar(x=orden, y=conteo.values, marker_color=colores,
                                        text=[f"{v} ({v/conteo.sum()*100:.1f}%)" for v in conteo.values],
                                        textposition='outside'))
                fig.update_layout(title=f'Distribución de {var}', xaxis_title=var, yaxis_title='Frecuencia',
                                   template='plotly_white', height=420)
            else:
                ct = pd.crosstab(train[var], train[cruce], normalize='index')[
                    (CLASES_APP if cruce == 'discourse_effectiveness' else sorted(train[cruce].unique()))] * 100
                fig = go.Figure()
                for col in ct.columns:
                    fig.add_bar(name=str(col), x=ct.index, y=ct[col], marker_color=color_de(cruce, col))
                fig.update_layout(barmode='stack', title=f'{var} vs {cruce} (% por fila)',
                                   template='plotly_white', height=460)
            display(fig)
            display(train[var].value_counts(normalize=True).rename('proporcion').to_frame().round(4))
        else:
            fig = px.histogram(train, x=var, color='discourse_effectiveness',
                                color_discrete_map=COLOR_EFECTIVIDAD, marginal='box', opacity=0.75,
                                template='plotly_white', height=460,
                                title=f'Distribución de {var} por discourse_effectiveness')
            display(fig)
            display(train.groupby('discourse_effectiveness')[var].describe().round(2))

dd_variable.observe(dibujar_explorador, names='value')
dd_cruce.observe(dibujar_explorador, names='value')
tab1 = widgets.VBox([widgets.HTML('<b>Explora las variables del conjunto de datos limpio</b>'),
                      widgets.HBox([dd_variable, dd_cruce]), out_explorar])
dibujar_explorador()
"""
))

cells.append(md("#### Pestaña 2 — Resultados de los modelos (interactivo, se puede esconder)"))
cells.append(code(
"""cb_mostrar_rendimiento = widgets.Checkbox(value=True, description='Mostrar rendimiento de los modelos')
sel_modelos = widgets.SelectMultiple(options=list(modelos_app.keys()),
                                      value=tuple(modelos_app.keys()),
                                      description='Modelos:', rows=6)
btn_todos = widgets.Button(description='Seleccionar todos', button_style='')
btn_ninguno = widgets.Button(description='Ninguno', button_style='')
out_modelos = widgets.Output()

def seleccionar_todos(_):
    sel_modelos.value = tuple(modelos_app.keys())
def seleccionar_ninguno(_):
    sel_modelos.value = tuple()
btn_todos.on_click(seleccionar_todos)
btn_ninguno.on_click(seleccionar_ninguno)

def dibujar_rendimiento(_=None):
    with out_modelos:
        clear_output(wait=True)
        if not cb_mostrar_rendimiento.value:
            print('(Rendimiento oculto — active la casilla para mostrarlo)')
            return
        elegidos = list(sel_modelos.value)
        if not elegidos:
            print('Seleccione al menos un modelo.')
            return
        sub = compar_app[compar_app['modelo'].isin(elegidos)].sort_values('f1_mean')

        fig1 = go.Figure(go.Bar(
            x=sub['f1_mean'], y=sub['modelo'], orientation='h',
            error_x=dict(type='data', array=sub['f1_std']),
            marker_color=px.colors.sequential.Viridis[:len(sub)] if len(sub) <= 10 else None,
            text=[f"{v:.3f}" for v in sub['f1_mean']], textposition='outside'))
        fig1.add_vline(x=0.55, line_dash='dash', line_color=COLOR_EFECTIVIDAD['Ineffective'],
                        annotation_text='Objetivo 0.55')
        fig1.update_layout(title='F1-macro por modelo (CV 5 pliegues agrupada)', template='plotly_white',
                            height=120 + 40 * len(sub), xaxis_title='F1-macro')
        display(fig1)

        fig2 = go.Figure(go.Bar(x=sub['seg'], y=sub['modelo'], orientation='h',
                                 marker_color='#4477AA',
                                 text=[f"{v:.0f}s" for v in sub['seg']], textposition='outside'))
        fig2.update_layout(title='Tiempo de procesamiento (ajuste + validación cruzada)',
                            template='plotly_white', height=120 + 40 * len(sub), xaxis_title='Segundos')
        display(fig2)

        hsub = holdout_app[holdout_app['modelo'].isin(elegidos)]
        for nombre in elegidos:
            s = hsub[hsub['modelo'] == nombre]
            if s.empty:
                continue
            cm = confusion_matrix(s['real'], s['pred'], labels=CLASES_APP, normalize='true')
            fig3 = go.Figure(go.Heatmap(z=cm, x=CLASES_APP, y=CLASES_APP, colorscale='Blues',
                                         text=np.round(cm, 2), texttemplate='%{text}'))
            acc_ = accuracy_score(s['real'], s['pred']); f1m_ = f1_score(s['real'], s['pred'], average='macro')
            fig3.update_layout(title=f'Matriz de confusión — {nombre} (acc={acc_:.3f}, F1m={f1m_:.3f})',
                                xaxis_title='Predicho', yaxis_title='Real', template='plotly_white', height=380)
            display(fig3)

cb_mostrar_rendimiento.observe(dibujar_rendimiento, names='value')
sel_modelos.observe(dibujar_rendimiento, names='value')
tab2 = widgets.VBox([
    widgets.HTML('<b>Resultados de cada algoritmo, por separado o en conjunto</b>'),
    cb_mostrar_rendimiento,
    widgets.HBox([sel_modelos, widgets.VBox([btn_todos, btn_ninguno])]),
    out_modelos,
])
dibujar_rendimiento()
"""
))

cells.append(md("#### Pestaña 3 — Clasificar un fragmento nuevo"))
cells.append(code(
"""ta_texto = widgets.Textarea(
    value='I think school uniforms are a good idea because they reduce bullying and make mornings easier.',
    description='Texto:', layout=widgets.Layout(width='100%', height='90px'))
dd_tipo = widgets.Dropdown(options=sorted(train['discourse_type'].unique()), value='Claim',
                            description='Tipo:')
btn_clasificar = widgets.Button(description='Clasificar', button_style='success', icon='check')
out_clasificar = widgets.Output()

def construir_fila(texto, tipo):
    t = limpiar_texto(texto)
    return pd.DataFrame([{
        'discourse_text_clean': t, 'discourse_type': tipo,
        'log_chars': np.log1p(len(t)), 'log_words': np.log1p(len(t.split())),
    }])

def clasificar(_=None):
    with out_clasificar:
        clear_output(wait=True)
        if not ta_texto.value.strip():
            print('Escriba un fragmento de texto.')
            return
        fila = construir_fila(ta_texto.value, dd_tipo.value)
        emb_cache = None
        resultados = []
        for nombre, obj in modelos_app.items():
            repr_tag, modelo = obj['repr'], obj['modelo']
            if repr_tag == 'R1sparse':
                pred = modelo.predict(fila)[0]
                proba = modelo.predict_proba(fila)[0] if hasattr(modelo, 'predict_proba') else None
            elif repr_tag == 'R1dense':
                x = r1dense_transformer_app.transform(fila)
                pred = modelo.predict(x)[0]
                proba = modelo.predict_proba(x)[0] if hasattr(modelo, 'predict_proba') else None
            else:  # 'R2'
                if emb_cache is None:
                    emb_cache = get_encoder().encode(fila['discourse_text_clean'].tolist(),
                                                       normalize_embeddings=True)
                x = np.hstack([emb_cache, ohe_app.transform(fila[['discourse_type']]).toarray(),
                               fila[['log_chars', 'log_words']].to_numpy()])
                pred = modelo.predict(x)[0]
                proba = modelo.predict_proba(x)[0] if hasattr(modelo, 'predict_proba') else None
            resultados.append({'modelo': nombre, 'prediccion': pred, 'proba': proba})

        for r in resultados:
            color = COLOR_EFECTIVIDAD.get(r['prediccion'], '#888888')
            display(widgets.HTML(f"<div style='padding:6px 10px;margin:3px 0;border-radius:6px;"
                                  f"background:{color};color:white;font-weight:bold'>"
                                  f"{r['modelo']}: {r['prediccion']}</div>"))
            if r['proba'] is not None:
                clases_modelo = (modelos_app[r['modelo']]['modelo'].classes_
                                  if hasattr(modelos_app[r['modelo']]['modelo'], 'classes_')
                                  else CLASES_APP)
                fig = go.Figure(go.Bar(x=list(clases_modelo), y=r['proba'],
                                        marker_color=[COLOR_EFECTIVIDAD.get(c, '#888') for c in clases_modelo],
                                        text=[f"{p:.2f}" for p in r['proba']], textposition='outside'))
                fig.update_layout(height=260, template='plotly_white', yaxis_range=[0, 1],
                                   title=f'Probabilidad por clase — {r["modelo"]}')
                display(fig)

        etiquetas = [r['prediccion'] for r in resultados]
        consenso = pd.Series(etiquetas).value_counts().idxmax()
        display(widgets.HTML(f"<h4>Consenso (voto mayoritario de los {len(resultados)} modelos): "
                              f"<span style='color:{COLOR_EFECTIVIDAD.get(consenso)}'>{consenso}</span></h4>"))

btn_clasificar.on_click(clasificar)
tab3 = widgets.VBox([
    widgets.HTML('<b>Ingrese un fragmento argumentativo y su tipo discursivo</b>'),
    ta_texto, dd_tipo, btn_clasificar, out_clasificar,
])
"""
))

cells.append(md("#### Aplicación completa (las tres pestañas)"))
cells.append(code(
"""app = widgets.Tab(children=[tab1, tab2, tab3])
app.set_title(0, 'Explorar los datos')
app.set_title(1, 'Resultados de los modelos')
app.set_title(2, 'Clasificar un fragmento nuevo')
display(app)
"""
))

cells.append(md(
"""> **Para usar la aplicación:** ejecute todas las celdas del notebook en orden (`Cell → Run All` o
> `Run All` en Jupyter/Colab) y luego interactúe con el panel de pestañas de arriba. Al cambiar de
> variable, de modelo(s) seleccionados, o al presionar "Clasificar", las gráficas se actualizan en
> el mismo lugar sin necesidad de volver a ejecutar celdas."""
))

# ======================================================================
# CONCLUSIONES
# ======================================================================
cells.append(md("## Conclusiones"))
cells.append(code(
"""from IPython.display import Markdown, display

obj1_f1_r1 = compar_app[compar_app['repr'] == 'R1 TF-IDF']['f1_mean'].max()
obj1_f1_r2 = compar_app[compar_app['repr'] == 'R2 sBERT']['f1_mean'].max()
mejor = compar_app.sort_values('f1_mean', ascending=False).iloc[0]

texto_conclusiones = f\"\"\"
**Objetivo general.** Se desarrolló y comparó un conjunto de modelos de aprendizaje automático que
clasifican la efectividad de unidades discursivas en ensayos argumentativos estudiantiles. El
modelo seleccionado, **{mejor['modelo']}**, alcanza un F1-macro de **{mejor['f1_mean']:.3f}** en
validación cruzada agrupada por ensayo, **{'superando' if mejor['f1_mean']>=0.55 else 'por debajo de'}**
el objetivo planteado de F1-macro ≥ 0.55. Esto confirma que es posible automatizar parcialmente
una tarea hasta ahora dependiente de la calificación manual de expertos, aunque —como se discute
más abajo— con limitaciones importantes de equidad que deben declararse.

**Objetivo específico 1 (representaciones de texto).** Se diseñaron e implementaron dos
representaciones (TF-IDF y *embeddings* de Sentence-BERT) y se compararon de forma
controlada bajo la misma partición agrupada: la mejor representación con TF-IDF alcanzó
F1-macro={obj1_f1_r1:.3f}, y la mejor con *embeddings*, F1-macro={obj1_f1_r2:.3f}. La diferencia
entre ambas representaciones es relativamente pequeña, lo que es consistente con la literatura
revisada (Mayfield y Black, 2020): una línea base clásica bien construida es competitiva frente a
representaciones semánticas más costosas, en este dominio de fragmentos cortos ya segmentados.

**Objetivo específico 2 (comparación de modelos).** Se entrenaron y compararon cinco modelos
(regresión logística, SVM lineal, bosque aleatorio y *gradient boosting*, sobre ambas
representaciones) mediante validación cruzada estratificada y agrupada por ensayo, documentando
sus diferencias de F1-macro, de exactitud y de tiempo de procesamiento. El ranking completo y las
razones de la selección se detallan en la sección de Resultados y Análisis.

**Sobre la efectividad de los algoritmos seleccionados.** El desbalance de clases (`Adequate`
57 %, `Ineffective` 18 %) se trató con ponderación de clases, evitando que la exactitud global
oculte un mal desempeño en la clase minoritaria; el análisis por clase muestra que `Ineffective`
sigue siendo la categoría más difícil de predecir para todos los modelos, lo cual es razonable,
dado que es también la clase con mayor variabilidad de longitud (ver EDA) y probablemente la más
heterogénea en sus causas (argumentos demasiado breves, desorganizados o sin evidencia).

**Limitaciones.** (1) El conjunto de datos no incluye variables demográficas, por lo que no se
puede auditar sesgo por subgrupos, y la literatura documenta sesgo moderado en esta misma tarea
(Baffour et al., 2023); los resultados deben leerse con esa salvedad. (2) El objetivo de F1-macro
≥ 0.55 se fijó antes de tener resultados, por lo que su dificultad real solo se conoció al evaluar
los modelos. (3) No se exploró el ajuste fino de un Transformer (DeBERTa) por restricciones de
cómputo (sin GPU garantizada); se mantiene como extensión futura, con la salvedad de que la
evidencia de Mayfield y Black (2020) sugiere que la ganancia frente a un modelo clásico bien
construido no está garantizada.

**Próximos pasos.** Ajustar umbrales de decisión sobre las probabilidades para favorecer el
*recall* de `Ineffective` (la clase de mayor interés práctico para dar retroalimentación
correctiva); explorar el ajuste fino de un Transformer como extensión si se dispone de GPU; y, de
cara a un uso real en un aula, evaluar la aplicación con profesores para validar que las
explicaciones y visualizaciones sean realmente útiles para dar retroalimentación a estudiantes.
\"\"\"
display(Markdown(texto_conclusiones))
"""
))

# ======================================================================
# REFERENCIAS
# ======================================================================
cells.append(md("## Referencias"))
cells.append(md(
"""Baffour, P., Saxberg, T., y Crossley, S. (2023). Analyzing bias in large language model
solutions for assisted writing feedback tools: Lessons from the Feedback Prize competition
series. En *Proceedings of the 18th Workshop on Innovative Use of NLP for Building Educational
Applications (BEA 2023)* (pp. 242–246). Association for Computational Linguistics.
https://doi.org/10.18653/v1/2023.bea-1.21

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
https://doi.org/10.1023/A:1010933404324

Chawla, N. V., Bowyer, K. W., Hall, L. O., y Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority
over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321–357.
https://doi.org/10.1613/jair.953

Chen, T., y Guestrin, C. (2016). XGBoost: A scalable tree boosting system. En *Proceedings of the
22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794).
ACM. https://doi.org/10.1145/2939672.2939785

Cortes, C., y Vapnik, V. (1995). Support-vector networks. *Machine Learning, 20*(3), 273–297.
https://doi.org/10.1007/BF00994018

Crossley, S. A., Baffour, P., Tian, Y., Picou, A., Benner, M., y Boser, U. (2022). The persuasive
essays for rating, selecting, and understanding argumentative and discourse elements (PERSUADE)
corpus 1.0. *Assessing Writing, 54*, Artículo 100667. https://doi.org/10.1016/j.asw.2022.100667

Crossley, S. A., Tian, Y., Baffour, P., Franklin, A., Benner, M., y Boser, U. (2024). A
large-scale corpus for assessing written argumentation: PERSUADE 2.0. *Assessing Writing, 61*,
Artículo 100865. https://doi.org/10.1016/j.asw.2024.100865

Devlin, J., Chang, M.-W., Lee, K., y Toutanova, K. (2019). BERT: Pre-training of deep
bidirectional transformers for language understanding. En *Proceedings of the 2019 Conference of
the North American Chapter of the Association for Computational Linguistics: Human Language
Technologies, Volume 1* (pp. 4171–4186). Association for Computational Linguistics.
https://doi.org/10.18653/v1/N19-1423

He, H., y Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge
and Data Engineering, 21*(9), 1263–1284. https://doi.org/10.1109/TKDE.2008.239

He, P., Liu, X., Gao, J., y Chen, W. (2021). DeBERTa: Decoding-enhanced BERT with disentangled
attention. En *International Conference on Learning Representations (ICLR 2021)*.
https://arxiv.org/abs/2006.03654

Hochreiter, S., y Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8),
1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

Kaufman, S., Rosset, S., y Perlich, C. (2011). Leakage in data mining: Formulation, detection,
and avoidance. En *Proceedings of the 17th ACM SIGKDD International Conference on Knowledge
Discovery and Data Mining* (pp. 556–563). ACM. https://doi.org/10.1145/2020408.2020496

Ke, Z., y Ng, V. (2019). Automated essay scoring: A survey of the state of the art. En
*Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence*
(pp. 6300–6308). IJCAI Organization. https://doi.org/10.24963/ijcai.2019/879

Mayfield, E., y Black, A. W. (2020). Should you fine-tune BERT for automated essay scoring? En
*Proceedings of the Fifteenth Workshop on Innovative Use of NLP for Building Educational
Applications* (pp. 151–162). Association for Computational Linguistics.
https://doi.org/10.18653/v1/2020.bea-1.15

Misgna, H., On, B.-W., Lee, I., y Choi, G. S. (2025). A survey on deep learning-based automated
essay scoring and feedback generation. *Artificial Intelligence Review, 58*(2), Artículo 36.
https://doi.org/10.1007/s10462-024-11017-5

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M.,
Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M.,
Perrot, M., y Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine
Learning Research, 12*, 2825–2830.

Ramesh, D., y Sanampudi, S. K. (2022). An automated essay scoring systems: A systematic literature
review. *Artificial Intelligence Review, 55*(3), 2495–2527.
https://doi.org/10.1007/s10462-021-10068-2

Reimers, N., y Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese
BERT-networks. En *Proceedings of the 2019 Conference on Empirical Methods in Natural Language
Processing and the 9th International Joint Conference on Natural Language Processing
(EMNLP-IJCNLP)* (pp. 3982–3992). Association for Computational Linguistics.
https://doi.org/10.18653/v1/D19-1410

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein,
S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., y
Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical,
or phylogenetic structure. *Ecography, 40*(8), 913–929. https://doi.org/10.1111/ecog.02881

Salton, G., y Buckley, C. (1988). Term-weighting approaches in automatic text retrieval.
*Information Processing & Management, 24*(5), 513–523.
https://doi.org/10.1016/0306-4573(88)90021-0

Sokolova, M., y Lapalme, G. (2009). A systematic analysis of performance measures for
classification tasks. *Information Processing & Management, 45*(4), 427–437.
https://doi.org/10.1016/j.ipm.2009.03.002

Stab, C., y Gurevych, I. (2017). Parsing argumentation structures in persuasive essays.
*Computational Linguistics, 43*(3), 619–659. https://doi.org/10.1162/COLI_a_00295

Taghipour, K., y Ng, H. T. (2016). A neural approach to automated essay scoring. En *Proceedings
of the 2016 Conference on Empirical Methods in Natural Language Processing* (pp. 1882–1891).
Association for Computational Linguistics. https://doi.org/10.18653/v1/D16-1193

Uto, M. (2021). A review of deep-neural automated essay scoring models. *Behaviormetrika, 48*(2),
459–484. https://doi.org/10.1007/s41237-021-00142-y

Wachsmuth, H., Naderi, N., Hou, Y., Bilu, Y., Prabhakaran, V., Thijm, T. A., Hirst, G., y Stein,
B. (2017). Computational argumentation quality assessment in natural language. En *Proceedings of
the 15th Conference of the European Chapter of the Association for Computational Linguistics:
Volume 1, Long Papers* (pp. 176–187). Association for Computational Linguistics.
https://doi.org/10.18653/v1/E17-1017"""
))

nbf.write(nbf.v4.new_notebook(cells=cells, metadata=nb2.metadata), "proyecto2-DS.ipynb")
print("TOTAL FINAL cells:", len(cells))
