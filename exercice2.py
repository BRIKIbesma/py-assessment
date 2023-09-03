import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# Convertir l'image croppée en niveaux de gris
cropped_img_grayscale = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

# Tracez un histogramme par bande spectrale RGB des valeurs pixels (installer matplolib pour les visualiser) avec numpy ou opencv
hist_channels = [cv2.calcHist([cropped_img], [i], None, [256], [0, 256]) for i in range(3)]


# Analysez les histogrammes pour segmenter le chateau avec un seuil
plt.figure()
for i, color in enumerate(['Blue', 'Green', 'Red']):
    plt.plot(hist_channels[i], color=color, label=f'Canal {color}')
plt.legend()
plt.title('Histogrammes des canaux RGB')

# Configurez le seuil pour la segmentation (vous pouvez ajuster ce seuil)
seuil = 100

# Appliquez le seuillage à l'image en niveaux de gris pour la segmenter
segmented_img = cv2.threshold(cropped_img_grayscale, seuil, 255, cv2.THRESH_BINARY)[1]

# Enregistrez vos resultats

