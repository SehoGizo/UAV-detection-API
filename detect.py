import matplotlib
matplotlib.use("TkAgg")
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ✅ Modeli yükle
model = YOLO("yolov8n.pt")

# ✅ Görüntüyü oku
image_path = "pexels-flo-dnd-989753-2100075.jpg"
image = cv2.imread(image_path)

# ✅ Modeli görüntü üzerinde çalıştır
results = model(image)

# ✅ Çıktıyı al ve ekranda göster
for r in results:
    img = r.plot()  # Çıktıyı al

    # 🔹 OpenCV yerine Matplotlib ile göster
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")  # Eksenleri kaldır
    plt.show()
