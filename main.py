import cv2
import numpy as np

# carga la imagen a analizar
imagen = cv2.imread('/mnt/data/circunsferencia.jpg')

# la pasa a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# aplica un desenfoque para reducir el ruido
gris_suavizado = cv2.GaussianBlur(gris, (9, 9), 2)

# implementa la transformada de Hough para detectar el círculo
circulos = cv2.HoughCircles(
    gris_suavizado,
    cv2.HOUGH_GRADIENT,
    dp=1.2,  # es la resolucion inversa del acumulado de la trasnformada
    minDist=20,  # distancia mínima entre los centros de los círculos detectados
    param1=50,  # umbral del detector de bordes en la parte superior
    param2=30,  # umbral del acumulador usado para la detección de círculos
    minRadius=10,  # radio mínimo del círculo 
    maxRadius=100  # radio máximo del círculo 
)

# dibuja los círculos detectados en la imagen que se subio en  primer instancia. 
if circulos is not None:
    circulos = np.uint16(np.around(circulos))
    for i in circulos[0, :]:
        # dibuja el contorno del círculo
        cv2.circle(imagen, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # dibuja el centro del círculo
        cv2.circle(imagen, (i[0], i[1]), 2, (0, 0, 255), 3)

# muestra la imagen con los círculos detectados
cv2.imshow('Detección de círculos con Transformada de Hough', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()
