import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request
from PIL import Image
import base64
from io import BytesIO
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Cargar modelos
modelo_datos = joblib.load('model/modelo_datos.pkl')
modelo_imagenes = load_model('model/modelo_imagen.keras')

# Cargar datos
df = pd.read_csv('data/merged_dummies.csv')
features = ['relapse', 'tumor_size', 'early_detection', 'sex', 'inflammatory_bowel_disease']

@app.route('/', methods=['GET', 'POST'])
def index():
    paciente_dict = None
    imagen_b64 = None
    confidence_final = None
    user_id = None
    confidence_datos = None
    confidence_imagen = None
    prediccion_datos = None
    prediccion_imagen = None

    if request.method == 'POST':
        user_id = int(request.form['user_id'])
        paciente = df[df['id'] == user_id]

        if not paciente.empty:
            paciente_info = paciente[features].copy()
            paciente_info['sex'] = paciente_info['sex'].map({0: 'F', 1: 'M'})
            for col in ['relapse', 'early_detection', 'inflammatory_bowel_disease']:
                paciente_info[col] = paciente_info[col].map({0: 'No', 1: 'Yes'})

            input_df = paciente[features]

            # 1. Predicción modelo datos
            prob_datos = modelo_datos.predict_proba(input_df)[0][1]  # probabilidad de benigno

            if prob_datos >= 0.5:
                prediccion_datos = "✅ Benigno"
                confidence_datos = prob_datos
            else:
                prediccion_datos = "⚠️ Maligno"
                confidence_datos = 1 - prob_datos

            # 2. Predicción modelo imágenes
            image_name = paciente['image_name'].values[0]

            # Buscar primero en Benigno, luego en Maligno
            image_path_benigno = os.path.join('Image/Benigno', f'{image_name}.jpeg')
            image_path_maligno = os.path.join('Image/Maligno', f'{image_name}.jpeg')

            if os.path.exists(image_path_benigno):
                image_path = image_path_benigno
            elif os.path.exists(image_path_maligno):
                image_path = image_path_maligno
            else:
                image_path = None

            if image_path:
                img = Image.open(image_path).resize((224, 224))
                img = np.array(img) / 255.0
                img = np.expand_dims(img, axis=0)
                pred_imagen_raw = modelo_imagenes.predict(img)
                confidence_imagen = pred_imagen_raw[0][0]

                # CORRECCIÓN: ajustar la confianza a ser benigno
                confidence_imagen = 1 - confidence_imagen

                # Convertir imagen a base64 para mostrarla en la web
                buffered = BytesIO()
                Image.open(image_path).save(buffered, format="JPEG")
                imagen_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            else:
                confidence_imagen = 0.5

            # 3. Combinar
            confidence_final = (confidence_datos + confidence_imagen) / 2

            # 4. Convertir datos a dict
            paciente_dict = paciente_info.iloc[0].to_dict()

    return render_template('index.html',
                           paciente=paciente_dict,
                           imagen_b64=imagen_b64,
                           confidence_final=confidence_final,
                           confidence_datos=confidence_datos,
                           confidence_imagen=confidence_imagen,
                           prediccion_datos=prediccion_datos,
                           prediccion_imagen=prediccion_imagen,
                           user_id=user_id)

if __name__ == "__main__":
    app.run(debug=True)