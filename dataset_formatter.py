import os
import shutil
import random

# Gerekli klasörleri oluştur
SOURCE_DIR = "C:/Coding/Projects/UAV-Detection/drone_dataset_yolo/dataset_txt"
os.makedirs(f"{SOURCE_DIR}/images/train", exist_ok=True)
os.makedirs(f"{SOURCE_DIR}/images/val", exist_ok=True)
os.makedirs(f"{SOURCE_DIR}/labels/train", exist_ok=True)
os.makedirs(f"{SOURCE_DIR}/labels/val", exist_ok=True)

# Görüntü dosyalarını listele
image_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".jpg")]
random.shuffle(image_files)

# %80 eğitim, %20 doğrulama olarak ayır
train_size = int(0.8 * len(image_files))

for i, file in enumerate(image_files):
    img_path = os.path.join(SOURCE_DIR, file)
    txt_path = img_path.replace(".jpg", ".txt")  # Etiket dosyası

    if i < train_size:
        shutil.move(img_path, f"{SOURCE_DIR}/images/train/")
        shutil.move(txt_path, f"{SOURCE_DIR}/labels/train/")
    else:
        shutil.move(img_path, f"{SOURCE_DIR}/images/val/")
        shutil.move(txt_path, f"{SOURCE_DIR}/labels/val/")

print("✅ Veri seti başarıyla düzenlendi ve eğitim için ayrıldı!")
