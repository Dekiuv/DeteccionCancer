# 🧬 Predicción de Cáncer de Colon

Este proyecto utiliza **datos clínicos** e **imágenes médicas** para predecir la probabilidad de que un tumor de colon sea **benigno** o **maligno**. La solución combina modelos de Machine Learning y Deep Learning en una **interfaz web** sencilla.

---

## 📂 Estructura del Proyecto

```plaintext
Predicción de Cáncer de Colon/
├── data_exploring.ipynb       # Exploración de datos y limpieza
├── EntrenoDatos.py            # Entreno datos clínicos
├── EntrenoImage.py            # Entreno imagenes
├── app.py                     # Backend aplicación
│
├── static/                    # Carpeta con complementos página web
│   ├── css/
│   ├── img/
│   └── js/
│
├── templates/                 # Estructura HTML
│   └── index.html
│
├── Image/                     # Carpeta con imagenes de colon
│   ├── Benigno/
│   └── Maligno/
│
├── modelos_por_clusters/      # Carpeta con modelos enternados
│   ├── modelo_datos.pkl
│   └── modelo_imagenes.keres
│
├── data/                      # Carpeta con los CSV
│   ├── analisis_cancer.csv
│   ├── historial_medico_imagenes.csv
│   ├── historial_medico.csv
│   ├── merged_dummies.csv
│   └── merged.csv
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
