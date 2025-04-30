# 🧬 Predicción de Cáncer de Colon

Este proyecto utiliza **datos clínicos** e **imágenes médicas** para predecir la probabilidad de que un tumor de colon sea **benigno** o **maligno**. La solución combina modelos de Machine Learning y Deep Learning en una **interfaz web** sencilla.

---

## 📂 Estructura del Proyecto

```plaintext
SistemaRecomendacion/
├── data_loader.py             # Carga de los CSV
├── Creadorcluster.csv         # Segmentación de usuarios por K-Means
├── Codo+grafico.csv           # Visualizar grafico codo y distribución de usuarios en clusters
├── entreno.py                 # Entrena un modelo SVD por cada cluster
├── ver_metricas_modelos.py    # Muestra las métricas (accuracy, precision, recall, F1) de cada modelo entrenado.
├── recomendador.py            # Recomendación para un usuario y cluster.
├── nlp.py                     # Permite buscar productos similares usando procesamiento de lenguaje natural (TF-IDF).
├── market_basket.py           # Reglas de asociación (Apriori)
├── app.py                     # Backend Flask principal
├── index.html                 # Interfaz HTML principal
├── styles.css                 # Estilos CSS
├── script.js                  # Lógica del frontend en JavaScript
│
├── Image/                     # Carpeta con imagenes de la página web
│   ├── MAPA.png
│   ├── github.png
│   └── supermercado.png
│
├── modelos_por_clusters/      # Carpeta con modelos enternados
│   ├── modelo_svd_cluster0.pkl
│   ├── modelo_svd_cluster1.pkl
│   ├── modelo_svd_cluster2.pkl
│   └── modelo_svd_cluster3.pkl
│
├── data/                      # Carpeta con los CSV
│   ├── Aisles.csv
│   ├── departments.csv
│   ├── order_products__prior.csv
│   ├── order_products__train.csv
│   ├── orders_cleaned.csv
│   └── products.csv
│
├── .gitignore                 # Archivos y carpetas ignoradas por Git
└── README.md                  # Documentación del proyecto
```

---

## 🚀 ¿Cómo funciona?

1. El usuario ingresa el **ID del paciente** (entre 1 y 10,000).
2. El sistema busca:
   - Sus datos clínicos (`relapse`, `tumor_size`, `early_detection`, `sex`, `inflammatory_bowel_disease`).
   - Su imagen médica (tumor de colon).
3. Se realizan dos predicciones:
   - Modelo de datos clínicos (`modelo_datos.pkl`).
   - Modelo de imágenes médicas (`modelo_imagen.keras`).
4. Se combinan las predicciones para calcular una **confianza final**.
5. Se muestra:
   - Informe médico del paciente.
   - Imagen médica.
   - Resultados individuales de cada modelo.
   - Barra de progreso que indica el % de probabilidad de ser benigno.

---

## 📈 Modelos utilizados

- **Modelo de Datos Clínicos:**  
  Entrenado mediante un clasificador Random Forest utilizando variables médicas seleccionadas (`EntrenoDatos.py`).

- **Modelo de Imágenes Médicas:**  
  Basado en **MobileNetV2** (`EntrenoImage.py`), con entrenamiento sobre imágenes categorizadas como `Benigno` o `Maligno`.

---

## 🛠 Tecnologías

- **Python 3.11**
- **Flask** para la web
- **TensorFlow / Keras** para el modelo de imágenes
- **Scikit-learn** para el modelo de datos
- **Pandas**, **NumPy**, **Matplotlib**
- **HTML5, CSS3** para la interfaz

---

## 📋 Requisitos de Instalación

1. Clona el repositorio:

```bash
git clone https://github.com/Dekiuv/DeteccionCancer
cd DeteccionCancer
```
2. Instalar librerias necesarias:
```bash
pip install -r requirements.txt
```
3. Ejecutar el archivo app.py
```bash
python app.py
```
4. Abrir el navegador a esta url:
```bash
http://127.0.0.1:5000
```
