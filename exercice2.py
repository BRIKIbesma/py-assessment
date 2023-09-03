import cv2
import numpy as np

img_path = "resources/chateau.png"

# Chargez l'image
img = cv2.imread(img_path)

# Vérifiez si l'image a été chargée correctement
if img is None:
    raise Exception("Impossible de charger l'image. Assurez-vous que le chemin du fichier est correct.")

# Affichez la taille de l'image
height, width, _ = img.shape
print(f"Taille de l'image : {width}x{height}")

# Cropez l'image autour du chateau au dimension ci-dessous
x_top = 950
y_top = 450
height = 170
width = 450

cropped_img = img[y_top:y_top + height, x_top:x_top + width]

# Tracez un histogramme par bande spectrale RGB des valeurs pixels (installer matplolib pour les visualiser) avec numpy ou opencv

# Analysez les histogrammes pour segmenter le chateau avec un seuil

# Enregistrez vos resultats

