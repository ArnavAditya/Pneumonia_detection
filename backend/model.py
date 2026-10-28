import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models, layers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WEIGHTS_PATH = os.path.join(BASE_DIR, '..', 'models', 'vgg16_weights.weights.h5')

# 🔥 Rebuild architecture
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

# Build model
model.build((None, 224, 224, 3))

# 🔥 Load weights ONLY
model.load_weights(WEIGHTS_PATH)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    pred = model.predict(img_array, verbose=0)
    confidence = float(pred[0][0])
    label = "PNEUMONIA" if confidence > 0.5 else "NORMAL"

    return label, confidence