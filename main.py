#IMPORTACION LIBRERIAS
from ultralytics import YOLO
import cv2


#Se lee modelo
model = YOLO("best7.pt")

#Camara
cap = cv2.VideoCapture(2)

#Bucle
while True:
    #Leer fotogramas
    ret, frame = cap.read()

    #Resultados red neuronal
    result = model.predict(frame, imgsz = 640)

    #Mostrar resultados
    anot = result[0].plot()

    #Mostrar fotogramas
    cv2.imshow("DETECCION OBJETOS", anot)

    #Fin Programa
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()



