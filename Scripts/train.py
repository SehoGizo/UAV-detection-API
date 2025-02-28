from ultralytics import YOLO

# YOLOv8 modelini yükle
model = YOLO("yolov8n.pt")
# Modeli eğit
if __name__ == "__main__":
    model.train(data="drone_dataset_yolo/dataset_txt/data.yaml",
                epochs=100,
                batch=32,
                imgsz=640,
                optimizer="Adam",
                device="cuda")  # GPU kullanımı için "cuda", CPU için "cpu" yaz

# Eğitilen modeli kaydet
model.export(format="onnx")  # ONNX formatında kaydetmek opsiyoneldir
