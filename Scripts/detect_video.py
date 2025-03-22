from ultralytics import YOLO
import cv2

# Eğitilmiş modeli yükle
model = YOLO("../runs/detect/train10/weights/best.pt")

# Video kaynağını belirle
video_path = "../istockphoto-1810266408-640_adpp_is.mp4"  # Buraya test etmek istediğin video yolunu gir
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    #  Modeli video kareleri üzerinde çalıştır
    results = model(frame)

    #  Sonucu ekranda göster
    for r in results:
        img = r.plot()
        cv2.imshow("Drone Detection", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
