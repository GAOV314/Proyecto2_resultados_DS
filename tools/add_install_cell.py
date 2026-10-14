import nbformat

nb = nbformat.read("proyecto2-DS.ipynb", as_version=4)

# Elimina cualquier intento previo corrupto (si las dos primeras celdas son el markdown/code de instalacion)
while len(nb.cells) >= 1 and nb.cells[0].cell_type == 'markdown' and 'Instalación de librerías' in nb.cells[0].source:
    del nb.cells[0]
while len(nb.cells) >= 1 and nb.cells[0].cell_type == 'code' and '%pip install' in nb.cells[0].source:
    del nb.cells[0]

install_md = (
    "## Instalación de librerías\n\n"
    "Ejecute la siguiente celda **una sola vez** al abrir el notebook (en Colab, Kaggle o localmente) "
    "para instalar todas las dependencias usadas: `pandas`, `numpy`, `scikit-learn`, `scipy`, "
    "`matplotlib`, `seaborn`, `joblib`, `ipywidgets`, `plotly` y `sentence-transformers` "
    "(esta última instala también `torch`, usado para los *embeddings* de Sentence-BERT). "
    "Si ya tiene un entorno con estas librerías instaladas (por ejemplo, el `venv` del repositorio), "
    "puede omitir esta celda sin problema."
)

install_code = (
    "%%capture\n"
    "# Instalación de dependencias del proyecto (seguro de re-ejecutar: pip omite lo ya instalado).\n"
    "# En Google Colab / Kaggle, esta celda deja el entorno listo para correr todo el notebook.\n"
    "%pip install -q pandas numpy scikit-learn scipy matplotlib seaborn joblib ipywidgets plotly sentence-transformers\n"
)

nb.cells.insert(0, nbformat.v4.new_code_cell(install_code))
nb.cells.insert(0, nbformat.v4.new_markdown_cell(install_md))

nbformat.write(nb, "proyecto2-DS.ipynb")
print("total cells now:", len(nb.cells))
print("--- cell 0 ---")
print(nb.cells[0].source)
print("--- cell 1 ---")
print(nb.cells[1].source)
