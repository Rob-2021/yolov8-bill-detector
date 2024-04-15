from ultralytics import YOLO
import cv2

# leemos nuestro modelo
model = YOLO('D:\proyectoIA3\yolov8-bill-detector\\runs\segment\\train\weights\\best.pt')

# realizamos la videocaptura
cap = cv2.VideoCapture(0)

while True:
    # leemos los fotogramas
    ret, frame = cap.read()

    # leemos resultados
    resultados = model.predict(frame, imgsz = 640, conf = 0.95)

    # mostramos los resultados
    anotaciones = resultados[0].plot()

    # mostramos los fotogramas
    cv2.imshow('frame', anotaciones)

    # cerrar la ventana
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()    