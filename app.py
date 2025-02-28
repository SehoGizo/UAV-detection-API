from fastapi import FastAPI, UploadFile, File
import uvicorn
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os

app = FastAPI()

# Modeli yükle
model = YOLO("runs/detect/train10/weights/best.pt")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Dosyayı geçici bir yere kaydet
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await file.read())
        temp_video_path = temp_video.name

    # Video okuma
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        return {"error": "Video yüklenemedi"}

    detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Video bittiğinde dur

        # YOLO modelini çalıştır
        results = model(frame)

        for r in results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = box.tolist()
                detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    cap.release()
    os.remove(temp_video_path)  # Geçici dosyayı sil

    return {"detections": detections}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
