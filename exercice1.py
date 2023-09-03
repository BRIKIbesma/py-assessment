import cv2
import numpy as np

# Charger une image binaire de ce format
IMG_HEIGHT = 480

IMG_WIDTH = 640
CHANNEL = 1 # Grayscale

with open("resources/thermal_img_640x480_16bits_grayscale.bin", 'rb') as file:
    buffer = file.read()

# Convertir le buffer en une image numpy
thermal_img = np.frombuffer(buffer, dtype=np.uint16).reshape((IMG_HEIGHT, IMG_WIDTH, CHANNEL))

# Faire une normalisation MIN/MAX sur 8 bits de l'image 
normalized_img = cv2.normalize(thermal_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Faites une normalisation MIN/MAX sur 8 bits de l'image (utilisez OpenCV)
min_val = np.min(thermal_img)
max_val = np.max(thermal_img)
normalized_img = ((thermal_img - min_val) / (max_val - min_val) * 255).astype(np.uint8) 


# Faites une normalisation MIN/MAX sur 8 bits de l'image (utiliser opencv)
normalized_img = 

# Enregistrer l'image normalisée au format png
