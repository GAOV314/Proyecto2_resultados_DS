"""
Construye proyecto2-DS.ipynb completo (Proyecto 2 - Resultados) a partir del notebook
existente (EDA, cells 0-54) + nuevas secciones de informe + modelado con tuning +
guardado de artefactos + aplicación interactiva (ipywidgets + Plotly).

Uso: python tools/build_notebook.py
"""
import nbformat as nbf
import copy

SRC = "proyecto2-DS_eda_base.ipynb"   # snapshot congelado del notebook original (cells 0-54)
OUT = "proyecto2-DS.ipynb"

nb_old = nbf.read(SRC, as_version=4)
base_cells = [copy.deepcopy(c) for c in nb_old.cells[0:55]]  # 0..54 inclusive: EDA completo

def md(src):
    return nbf.v4.new_markdown_cell(src)

def code(src):
    return nbf.v4.new_code_cell(src)

cells = []

# ------------------------------------------------------------------
# Celda de instalación de dependencias, al inicio de todo (Colab/Kaggle/local).
cells.append(md(
"""## Instalación de librerías

Ejecute la siguiente celda **una sola vez** al abrir el notebook (en Colab, Kaggle o localmente)
para instalar todas las dependencias usadas: `pandas`, `numpy`, `scikit-learn`, `scipy`,
`matplotlib`, `seaborn`, `joblib`, `ipywidgets`, `plotly` y `sentence-transformers`
(esta última instala también `torch`, usado para los *embeddings* de Sentence-BERT).
Si ya tiene un entorno con estas librerías instaladas (por ejemplo, el `venv` del repositorio),
puede omitir esta celda sin problema."""
))
cells.append(code(
"""%%capture
# Instalación de dependencias del proyecto (seguro de re-ejecutar: pip omite lo ya instalado).
# En Google Colab / Kaggle, esta celda deja el entorno listo para correr todo el notebook.
%pip install -q pandas numpy scikit-learn scipy matplotlib seaborn joblib ipywidgets plotly sentence-transformers
"""
))

# ------------------------------------------------------------------
# 0-1: título + integrantes (ya en base_cells[0:2]) -> los dejamos, pero
# agregamos una nota de versión/alcance justo después.
cells.extend(base_cells[0:2])

cells.append(md(
"""> **Nota sobre esta entrega.** Este notebook integra el *Proyecto 2 — Resultados* completo:
> continúa el *Proyecto 1 — Análisis Exploratorio* (mismo reto, mismo dataset) y el documento de
> *Revisión bibliográfica y selección de algoritmos*. Contiene: Introducción, Objetivos, Marco
> Teórico, Metodología, Resultados y Análisis de Resultados (EDA, modelado, *tuning*, comparación
> de algoritmos y la aplicación interactiva), Conclusiones y Referencias (APA)."""
))

# ------------------------------------------------------------------
# Introducción (envuelve situación problemática + problema científico ya existentes)
cells.append(md("## Introducción"))
cells.append(md(
"""Este informe documenta la segunda entrega del Proyecto 2 del curso CC3084 — Data Science,
correspondiente al reto **"Trace the Ace"** (Procesamiento del Lenguaje Natural), basado en la
competencia de Kaggle *Feedback Prize – Predicting Effective Arguments* y el corpus **PERSUADE**
(Crossley et al., 2022, 2024). El documento reúne el trabajo realizado en las dos entregas previas
del grupo (análisis exploratorio y revisión bibliográfica con selección preliminar de algoritmos)
y lo extiende con la construcción, el ajuste de hiperparámetros, la evaluación comparativa de
varios modelos de aprendizaje automático, y una aplicación interactiva que permite explorar los
datos y clasificar nuevos fragmentos de texto con los modelos entrenados."""
))
cells.extend(base_cells[2:3])  # Situación problemática
cells.extend(base_cells[3:4])  # Problema científico

# ------------------------------------------------------------------
# Objetivos (ya existe en base_cells[4])
cells.extend(base_cells[4:5])

# ------------------------------------------------------------------
# Marco Teórico (NUEVO)
cells.append(md("## Marco Teórico"))

cells.append(md(
"""### Procesamiento del Lenguaje Natural (NLP)

El Procesamiento del Lenguaje Natural es el subcampo de la inteligencia artificial dedicado a que
los sistemas de cómputo analicen, interpreten y generen lenguaje humano escrito o hablado. Aplicado
a la escritura estudiantil, el NLP busca construir sistemas que asignen puntuaciones o categorías
de calidad a textos producidos por estudiantes (*automated writing evaluation*), evolucionando
desde modelos con características diseñadas manualmente (longitud, riqueza léxica, ortografía,
complejidad sintáctica) hacia representaciones distribuidas, redes neuronales y arquitecturas
Transformer (Ke y Ng, 2019; Misgna et al., 2025; Ramesh y Sanampudi, 2022; Uto, 2021). Una
distinción útil que estructura esta revisión es separar dos componentes de todo sistema de NLP:
la **representación** del texto (qué características se extraen) y el **algoritmo de predicción**
(cómo esas características se transforman en una clase o puntuación estimada) (Misgna et al., 2025).

En este proyecto, la tarea de NLP consiste en clasificar la efectividad argumentativa
(*Ineffective*, *Adequate*, *Effective*) de unidades discursivas ya segmentadas y tipadas
(`discourse_type`), por lo que no se requieren modelos de segmentación ni de minería de argumentos
(Stab y Gurevych, 2017); el problema se reduce a una clasificación de texto corto, con variables
auxiliares categóricas y numéricas.

### Actividades de preprocesamiento

El preprocesamiento de texto busca estandarizar el contenido crudo antes de representarlo
numéricamente. En este proyecto se aplicaron las siguientes actividades, varias de ellas
identificadas ya en el análisis exploratorio (Proyecto 1):

- **Limpieza de artefactos de codificación (*mojibake*)** y de espacios sobrantes al inicio/final
  del fragmento, producto de cómo se segmentó el texto original del ensayo.
- **Normalización de longitud**: derivar `len_chars` y `len_words`, y aplicar `log1p` para corregir
  la fuerte asimetría positiva observada (skew entre 1.1 y 3.8 según el grupo).
- **Vectorización por bolsa de palabras ponderada (TF-IDF)**: pondera cada término por su
  frecuencia en el documento y su rareza en el corpus (Salton y Buckley, 1988), capturando
  unigramas y bigramas.
- **Representación semántica mediante *embeddings* de oraciones**: Sentence-BERT (Reimers y
  Gurevych, 2019) produce vectores densos donde fragmentos de significado similar quedan
  próximos en el espacio vectorial, incluso si usan vocabulario distinto.
- **Codificación de variables categóricas auxiliares** (`discourse_type`) mediante *one-hot
  encoding*, y escalado logarítmico de las variables de longitud.
- **Partición agrupada de datos**: dado que los fragmentos de un mismo ensayo no son
  independientes, el particionamiento de entrenamiento/validación/prueba se agrupa por
  `essay_id` (Kaufman et al., 2011; Roberts et al., 2017), evitando fuga de información.

### Algoritmos de aprendizaje automático usados para NLP

Siguiendo la revisión bibliográfica del grupo (Revisión Bibliográfica y Selección de Algoritmos,
2026), se consideran dos familias de algoritmos:

**Clasificadores clásicos sobre representaciones dispersas o densas.**
- *Regresión logística multinomial*: modelo lineal probabilístico, rápido, interpretable y que
  admite ponderación de clases (`class_weight='balanced'`) para tratar el desbalance.
- *Máquinas de vectores de soporte (SVM) lineales* (Cortes y Vapnik, 1995): funcionan bien en
  espacios de alta dimensión y dispersos como el de TF-IDF.
- *Ensambles de árboles* (bosques aleatorios, Breiman, 2001; *gradient boosting*, Chen y Guestrin,
  2016): capturan interacciones no lineales entre el tipo discursivo, la longitud y el contenido
  que los modelos lineales no representan. En este proyecto se usa la implementación de
  *histogram-based gradient boosting* de scikit-learn (`HistGradientBoostingClassifier`), que
  ofrece el mismo principio algorítmico que XGBoost (Chen y Guestrin, 2016) sin depender de una
  biblioteca nativa adicional (OpenMP), lo que simplifica la reproducibilidad del entorno.

**Representaciones de texto.**
- *TF-IDF* (R1): eficiente, interpretable y con buen desempeño como línea base (Mayfield y Black,
  2020).
- *Embeddings de Sentence-BERT congelados* (`all-MiniLM-L6-v2`, R2): capturan similitud semántica
  que TF-IDF no recoge (Reimers y Gurevych, 2019), sin pagar el costo de ajustar finamente un
  Transformer completo.

**Extensión no explorada en esta entrega.** El ajuste fino de un Transformer tipo DeBERTa (He et
al., 2021) —empleado por las soluciones ganadoras de la competencia original (Baffour et al.,
2023)— se mantiene como extensión opcional, dado que Mayfield y Black (2020) muestran que el ajuste
fino de BERT no siempre supera a una línea base clásica bien construida, a un costo computacional
considerablemente mayor y sin GPU garantizada en el entorno del curso.

### Desbalance de clases y métricas de evaluación

La variable objetivo está desbalanceada (`Adequate` 57.06 %, `Effective` 25.37 %, `Ineffective`
17.58 %), por lo que la exactitud (*accuracy*) puede ser engañosa (He y Garcia, 2009). Se usa
**F1 macro** —que promedia el F1 de cada clase con el mismo peso (Sokolova y Lapalme, 2009)— como
métrica principal, con un objetivo de desempeño de F1 macro ≥ 0.55, complementada con F1 por
clase y matrices de confusión. El tratamiento del desbalance se realiza mediante ponderación de
clases (`class_weight='balanced'`), en línea con la estrategia a nivel de algoritmo descrita por
He y Garcia (2009)."""
))

# ------------------------------------------------------------------
# Metodología (NUEVO)
cells.append(md("## Metodología"))
cells.append(md(
"""### Pasos seguidos por el grupo

1. Revisión bibliográfica de algoritmos usados en evaluación automática de escritura y
   clasificación de texto (entrega previa).
2. Análisis exploratorio de los datos: calidad, estructura jerárquica, distribución de variables
   categóricas y numéricas, y su cruce (entrega previa, incluida en este notebook).
3. Selección preliminar de representaciones de texto y algoritmos, con justificación teórica
   (entrega previa).
4. Construcción de las representaciones R1 (TF-IDF + variables auxiliares) y R2 (Sentence-BERT +
   variables auxiliares).
5. Entrenamiento y ajuste de hiperparámetros de los modelos del núcleo (regresión logística, SVM
   lineal, *gradient boosting*) sobre ambas representaciones, con validación cruzada agrupada.
6. Evaluación y comparación de los modelos por efectividad (F1 macro, F1 por clase), tiempos de
   procesamiento y errores (matrices de confusión).
7. Selección del modelo final y guardado de los artefactos entrenados (vectorizador, codificador
   de variables, clasificador).
8. Construcción de una aplicación interactiva para explorar los datos y clasificar nuevos
   fragmentos con los modelos entrenados.

### Selección de los conjuntos de entrenamiento y prueba

Debido a que los 36,765 fragmentos provienen de solo 4,191 ensayos (cada ensayo aporta en
promedio 8-9 fragmentos), una partición aleatoria por fila dejaría fragmentos del mismo ensayo
en entrenamiento y en validación/prueba, lo que produce una fuga de información y estimaciones de
desempeño demasiado optimistas (Kaufman et al., 2011; Roberts et al., 2017). Por ello:

- La **validación durante el ajuste de hiperparámetros y la comparación de modelos** se realiza
  con `StratifiedGroupKFold` (5 pliegues), agrupando por `essay_id` y intentando preservar la
  proporción de clases en cada pliegue.
- Para el **detalle por clase, las matrices de confusión y la aplicación** se reserva, además, un
  conjunto de prueba (*hold-out*) del 20 % de los ensayos (`GroupShuffleSplit`), de modo que
  ningún ensayo del conjunto de prueba haya sido visto durante el entrenamiento del modelo final.
- Todo el preprocesamiento que depende de los datos (ajuste del vectorizador TF-IDF, de la
  reducción de dimensionalidad y de los codificadores) se ajusta **únicamente dentro de cada
  pliegue o dentro del conjunto de entrenamiento**, nunca sobre el conjunto completo antes de
  particionar.

### Selección de los algoritmos y razones

La selección de algoritmos (ver Marco Teórico y la revisión bibliográfica del grupo) se basó en
cinco criterios: (a) cumplir los objetivos específicos (al menos dos representaciones de texto y
tres modelos); (b) tener evidencia previa en tareas cercanas de evaluación automática de escritura;
(c) admitir ponderación de clases para tratar el desbalance; (d) ser viable con los recursos de
cómputo de un proyecto de curso (sin GPU garantizada); y (e) permitir incorporar `discourse_type`
y la longitud del fragmento, identificados en el EDA como predictores con asociación débil pero
significativa (V de Cramér = 0.127). Esto llevó al núcleo de tres familias de clasificadores
(regresión logística, SVM lineal, *gradient boosting*) evaluadas sobre dos representaciones
(TF-IDF y Sentence-BERT), descartando modelos de etiquetado de secuencias (CRF), Transformers de
contexto largo (Longformer) y redes recurrentes/convolucionales dedicadas, por no aportar ventaja
clara frente al costo en fragmentos cortos (mediana 28 palabras) ya segmentados y tipados.

### Selección de herramientas

**Recursos de cómputo.** Todo el proyecto se ejecutó en CPU (sin GPU dedicada), en un entorno
reproducible mediante un entorno virtual de Python (`venv`). Esta restricción de cómputo es,
precisamente, uno de los criterios que descartó el ajuste fino de un Transformer como punto de
partida (ver Marco Teórico).

**Lenguaje de programación.** Python 3.12.

**Bibliotecas y paquetes.**
- `pandas`, `numpy` — manipulación de datos.
- `scikit-learn` — vectorización TF-IDF, codificación categórica, reducción de dimensionalidad
  (SVD), modelos de clasificación, validación cruzada agrupada, búsqueda de hiperparámetros y
  métricas.
- `sentence-transformers` (`all-MiniLM-L6-v2`) — *embeddings* de oraciones congelados (R2).
- `scipy` — prueba chi-cuadrado de independencia (EDA categórico).
- `matplotlib`, `seaborn` — visualizaciones estáticas para el informe.
- `plotly` — visualizaciones interactivas dentro de la aplicación.
- `ipywidgets` — controles interactivos (pestañas, listas desplegables, casillas, botones) de la
  aplicación, embebidos en este mismo Jupyter Notebook.
- `joblib` — serialización de los artefactos del modelo final."""
))

# ------------------------------------------------------------------
# Resultados y Análisis de Resultados — header + datos + limpieza + EDA (ya existen)
cells.append(md("## Resultados y Análisis de Resultados"))
cells.append(md("### Características del conjunto de datos original"))
cells.extend(base_cells[5:16])    # descripción de datos (cells 5-15)
cells.extend(base_cells[16:23])   # limpieza y preprocesamiento (cells 16-22)
cells.append(md("### Análisis exploratorio de los datos"))
cells.extend(base_cells[23:55])   # EDA cuantitativo + categórico (cells 23-54)

nbf.write(nbf.v4.new_notebook(cells=cells, metadata=nb_old.metadata), "proyecto2-DS_part1.ipynb")
print("part1 cells:", len(cells))
