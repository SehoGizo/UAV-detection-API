import matplotlib
matplotlib.use("TkAgg")
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Modeli yükle
model = YOLO("../runs/detect/train10/weights/best.pt")

# Görüntüyü oku
image_path = "../drone_dataset_yolo/dataset_txt/images/train_augmented/0262.jpg"
image = cv2.imread(image_path)

# Modeli görüntü üzerinde çalıştır
results = model(image)

# Çıktıyı al ve ekranda göster
for r in results:
    img = r.plot()  # Çıktıyı al

    #  OpenCV yerine Matplotlib ile göster
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")  # Eksenleri kaldır
    plt.show()
