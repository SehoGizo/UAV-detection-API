from ultralytics import YOLO

# YOLOv8 modelini yükle
model = YOLO("yolov8n.pt")  # Küçük ve hızlı versiyon

# Modeli eğit
model.train(data="C:/Coding/Projects/UAV-Detection/dataset_txt/data.yaml",
            epochs=50, batch=8, imgsz=640)

# Eğitilmiş modeli kaydet
model_path = "C:/Coding/Projects/UAV-Detection/models/uav_detection_model.pt"
model.export(format="torchscript")
print(f"✅ Model kaydedildi: {model_path}")
