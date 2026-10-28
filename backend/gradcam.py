import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

def generate_gradcam(model, img_array, img_path, save_path):
    base_vgg = model.layers[0]
    last_conv_layer = base_vgg.get_layer('block5_conv3')

    x = base_vgg.output
    for layer in model.layers[1:]:
        x = layer(x)

    grad_model = Model(
        inputs=base_vgg.input,
        outputs=[last_conv_layer.output, x]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    cv2.imwrite(save_path, superimposed_img)

    return save_path