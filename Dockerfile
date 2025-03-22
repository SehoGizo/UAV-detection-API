# ✅ Resmi Python 3.12 görüntüsünü kullan
FROM python:3.12

# ✅ Çalışma dizinini ayarla
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0
# ✅ Gerekli kütüphaneleri yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Ana dosyaları kopyala
COPY app.py Scripts/detect.py Scripts/detect_video.py Scripts/train.py Scripts/data_augmentation.py ./
COPY ./runs/detect/train10/weights/best.pt /app/best.pt



# ✅ API'yi başlat
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
