# ================================
# 1. Importar librerías necesarias
# ================================
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ================================
# 2. Definir parámetros principales
# ================================
img_size = (224, 224)          # tamaño de las imágenes
batch_size = 32                # tamaño de lote
epochs = 10                    # número de épocas de entrenamiento
ruta_imagenes = 'Image'        # carpeta que contiene subcarpetas Benigno y Maligno

# ================================
# 3. Preparar generadores de datos (con augmentación simple)
# ================================
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # normalizar imágenes para MobileNetV2
    validation_split=0.2                      # 20% de validación
)

train_gen = datagen.flow_from_directory(
    ruta_imagenes,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    ruta_imagenes,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# ================================
# 4. Definir el modelo basado en MobileNetV2
# ================================

# Cargar MobileNetV2 preentrenada en ImageNet (sin la capa superior)
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # congelar las capas base para no sobreentrenar

# Añadir nuevas capas superiores
x = base_model.output
x = GlobalAveragePooling2D()(x)         # capa de pooling para reducir dimensiones
x = Dropout(0.3)(x)                      # dropout para prevenir overfitting
output = Dense(1, activation='sigmoid')(x)  # capa de salida binaria (benigno/maligno)

# Definir el modelo final
model = Model(inputs=base_model.input, outputs=output)

# Compilar el modelo
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ================================
# 5. Entrenamiento del modelo
# ================================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs
)

# ================================
# 6. Guardar el modelo entrenado
# ================================
model.save("modelo_imagen.keras")
print("✅ Modelo guardado como modelo_mobilenet_colon.keras")

# ================================
# 7. Graficar la precisión del entrenamiento y validación
# ================================
plt.plot(history.history['accuracy'], label='Precisión Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión Validación')
plt.title('Precisión del modelo')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)
plt.show()