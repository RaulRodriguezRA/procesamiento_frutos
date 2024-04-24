import numpy as np
import cv2
import sys

img = np.array([0, 0], np.uint8)
imagen_final = None

params = cv2.SimpleBlobDetector.Params()
params.filterByArea = True
params.minArea = 200
params.maxArea = sys.maxsize + 1


def limpieza(imagen, erosion, dilatacion, kblur):
    # Convertir a escala de grises
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    kernel_ero = np.ones(erosion, np.uint8)
    kernel_dil = np.ones(dilatacion, np.uint8)

    # Aplicar Dilatacion
    erosion = cv2.erode(gray, kernel_ero, iterations=1)

    # Aplicar blur
    blur = cv2.blur(erosion, kblur)

    # Aplicar Dilatacion
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV)
    dilatacion = cv2.dilate(mask, kernel_dil)
    return dilatacion


def calculo_hsv(h_prom, s_prom, v_prom, imagen, hsv):
    global img, anchura, altura, imagen_final
    img = imagen
    anchura, altura, _ = np.shape(imagen)
    imagen_final = np.array([anchura, altura], np.uint16)
    h, s, v = cv2.split(hsv)

    # Matrices Blanca y Negra
    white = np.full(h.shape + (3,), 255, np.uint8)
    black = np.full(h.shape + (3,), 0, np.uint8)

    h_condition = h <= h_prom + 5
    final_condition = h_condition
    print(final_condition)

    dilatacion = limpieza(
        # Matriz condicional
        np.where(np.sqrt((h_prom - h) ** 2 + (s_prom - s) ** 2 + (v_prom - v) ** 2)[..., None] <= 50, black, white),
        (5, 5),
        (31, 31),
        (21, 21)
    )

    cv2.imshow("tryme", np.where(cv2.cvtColor(dilatacion, cv2.COLOR_GRAY2BGR) == white, imagen, black))
    contours, hierarchy = cv2.findContours(dilatacion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    imagen_leyenda = cv2.putText(img, str(len(contours)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.imshow("Draw", cv2.drawContours(imagen_leyenda, contours, -1, (0, 255, 0), 3))
    cv2.imshow("Filtrado", dilatacion)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
