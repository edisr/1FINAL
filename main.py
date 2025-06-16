import cv2


def contar_huecos(region_binaria):
    # Invertir para que huecos sean blancos
    invertida = cv2.bitwise_not(region_binaria)

    # Detectar componentes conectados en la imagen invertida
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(invertida, connectivity=8)

    count = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 20:  # Solo huecos suficientemente grandes
            count += 1
    return count

def reconocer_jeroglifico(region_binaria):
    huecos = contar_huecos(region_binaria)

    if huecos == 0:
        return 'D'  # Djed
    elif huecos == 1:
        return 'A'  # Ankh
    elif huecos == 2:
        return 'K'  # Akhet
    elif huecos == 3:
        return 'S'  # Scarab
    elif huecos == 4:
        return 'W'  # Was
    elif huecos >= 5:
        return 'J'  # Wedjat
    return '?'

def procesar_imagen(path_imagen):
    original = cv2.imread(path_imagen)
    imagen = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    imagen_suave = cv2.GaussianBlur(imagen, (5,5), 0)
    _, binaria = cv2.threshold(imagen_suave, 127, 255, cv2.THRESH_BINARY_INV)

    # NO aplicamos erosión aquí para no dañar huecos reales
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)

    resultados = []
    imagen_resultado = original.copy()

    for i in range(1, num_labels):  # Ignorar fondo
        x, y, w, h, area = stats[i]
        if area < 150:  # Evitar falsos positivos
            continue

        region = binaria[y:y+h, x:x+w]
        letra = reconocer_jeroglifico(region)
        resultados.append(letra)

        # Dibujar letra y rectángulo
        cv2.rectangle(imagen_resultado, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(imagen_resultado, letra, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    resultados.sort()
    print("Letras reconocidas:", ''.join(resultados))
    cv2.imwrite("resultado.png", imagen_resultado)
    cv2.imshow("Clasificación", imagen_resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ejecutar
procesar_imagen("Ejemplo1.png")
