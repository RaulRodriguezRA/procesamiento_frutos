import cv2
import tkinter as tk
from PIL import ImageTk, Image
import deteccion as det


ruta_imagen = "Planta.png"
imgBGR = cv2.imread(ruta_imagen)
imgCV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
root = tk.Tk()
root.geometry()
capturas = 0
h = 0
s = 0
v = 0


def leftclick(event):
    global capturas, h, s, v
    capturas += 1
    print(imgCV[event.y, event.x])
    punto_h, punto_s, punto_v = imgCV[event.y, event.x]
    h += punto_h
    s += punto_s
    v += punto_v
    if capturas == 5:
        root.destroy()
        print(f"{h / capturas} {s / capturas} {v / capturas}")
        det.calculo_hsv(h / capturas, s / capturas, v / capturas, imgBGR, imgCV)


# import image
img = ImageTk.PhotoImage(Image.open(ruta_imagen))
panel = tk.Label(root, image=img)
panel.bind("<Button-1>", leftclick)
panel.pack(fill="both", expand=1)
root.mainloop()
