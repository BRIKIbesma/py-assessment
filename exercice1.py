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
 


# Faites une normalisation MIN/MAX sur 8 bits de l'image (utiliser opencv)
normalized_img = 

# Enregistrer l'image normalisée au format png
