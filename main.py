"""
@author: Samuel Torres y Edis Fernandez
"""
import cv2
import numpy as np
from skimage.measure import label
from scipy.ndimage import binary_dilation

def generate_marker(binary_image):
    inverted_image =cv2.bitwise_not(binary_image)
    # Crear la imagen marcador
    marker = np.zeros_like(binary_image)
    marker[0, :] = inverted_image[0, :]
    marker[-1, :] = inverted_image[-1, :]
    marker[:, 0] = inverted_image[:, 0]
    marker[:, -1] = inverted_image[:, -1]

    return marker

def morphological_reconstruction(marker, mask):
    if marker.shape != mask.shape:
        raise ValueError("Las dimensiones de marker y mask deben ser iguales.")

    reconstructed = marker.copy()

    while True:
        previous = reconstructed.copy()
        dilated = binary_dilation(reconstructed, structure=np.ones((3, 3)))
        reconstructed = dilated & mask
        if np.array_equal(reconstructed, previous):
            break

    return reconstructed

def rellenar(img):
    img_inverted = cv2.bitwise_not(img)
    marker = generate_marker(img)
    reconstructed = morphological_reconstruction(marker, img_inverted)

    return 255-reconstructed*255

def contar_huecos(agujeros):
    numero_agujeros = label((agujeros == 255).astype(np.uint8), connectivity=2)
    numero = np.max(numero_agujeros)

    return numero

def reconocer_jeroglifico(huecos):
    jeroglifico = ['W','A','K','J','S','D']
    if ( huecos < 0 or huecos > 6) :
        return '?'

    return jeroglifico[huecos]

def procesar_imagen(path_imagen):
    original = cv2.imread(path_imagen,0)
    _, mask_bin = cv2.threshold(original, 127, 255, cv2.THRESH_BINARY)
    resultado = []
    mask_bin_inverted = cv2.bitwise_not(mask_bin)
    labeled_image = label((mask_bin_inverted == 255).astype(np.uint8), connectivity=2)
    num_formas = np.max(labeled_image)
    for i in range(1,num_formas+1):
        figura = (labeled_image == i)
        figura = figura.astype(np.uint8) * 255

        agujeros_rellenados = rellenar(figura)
        agujeros = np.bitwise_xor(agujeros_rellenados, figura)
        numero_huecos = contar_huecos(agujeros)
        print(numero_huecos)
        resultado.append(reconocer_jeroglifico(numero_huecos))

    print(''.join(sorted(resultado)))





# Ejecutar
procesar_imagen("imagenes/Ejemplo2.png")
