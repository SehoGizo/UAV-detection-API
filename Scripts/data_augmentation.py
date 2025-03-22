import albumentations as A
import cv2
import os
import numpy as np


# YOLO formatındaki label dosyalarını işle
def load_yolo_labels(label_path):
    with open(label_path, "r") as f:
        labels = [line.strip().split() for line in f.readlines()]
    return labels


def save_yolo_labels(label_path, labels):
    with open(label_path, "w") as f:
        for label in labels:
            f.write(" ".join(map(str, label)) + "\n")


# Bounding box değerlerini sınırlandırma fonksiyonu
def clamp(value, min_value=0.0, max_value=1.0):
    """Bounding box değerlerini 0.0 - 1.0 aralığında tutar."""
    return max(min_value, min(value, max_value))


# YOLO Bounding Box'ları sınırlandırma fonksiyonu (Not in Bounds hatasını engeller)
def fix_bounding_box(x_center, y_center, width, height):
    """Eğer bounding box sınırların dışına çıkıyorsa, sınır içinde kalacak şekilde düzeltilir."""
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    # Eğer bounding box dışarı çıkıyorsa, sınır içinde tut
    x_min = clamp(x_min)
    y_min = clamp(y_min)
    x_max = clamp(x_max)
    y_max = clamp(y_max)

    #  Güncellenmiş koordinatlar
    new_x_center = (x_min + x_max) / 2
    new_y_center = (y_min + y_max) / 2
    new_width = x_max - x_min
    new_height = y_max - y_min

    return new_x_center, new_y_center, new_width, new_height


# Veri artırma işlemleri (YOLOv8 ile uyumlu olacak şekilde)
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussianBlur(p=0.2),
    A.Rotate(limit=15, p=0.5),  # Dönme işlemi bounding box'ları bozabilir, bunu düzelttik.
    A.Normalize()
], bbox_params=A.BboxParams(format="yolo", label_fields=["category_id"], min_visibility=0.2))


def augment_image_and_labels(image_path, label_path, output_image_path, output_label_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"⚠️ Uyarı: {image_path} dosyası okunamadı!")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV BGR formatında okur, RGB’ye çevir

    labels = load_yolo_labels(label_path)

    # Label formatını YOLO'dan Albumentations'a uygun hale getir
    boxes = []
    category_ids = []
    for label in labels:
        category_id = int(label[0])  # YOLO formatında ilk değer class_id'dir
        x_center, y_center, width, height = map(float, label[1:])  # [x_center, y_center, width, height]

        #  Eğer bounding box dışarı taşarsa, içeri al
        x_center, y_center, width, height = fix_bounding_box(x_center, y_center, width, height)

        boxes.append([x_center, y_center, width, height])
        category_ids.append(category_id)

    # Veri artırma uygula
    augmented = augmentation(image=image, bboxes=boxes, category_id=category_ids)

    #  Dönüştürülen görüntüyü tekrar OpenCV formatına çevir
    augmented_image = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)

    # OpenCV'nin desteklediği uint8 formatına çevir
    augmented_image = (augmented_image * 255).astype(np.uint8)

    # Güncellenmiş bounding box koordinatlarını sınırlandır
    updated_labels = []
    for bbox, category_id in zip(augmented["bboxes"], augmented["category_id"]):
        new_x_center, new_y_center, new_width, new_height = fix_bounding_box(*bbox)  # Koordinatları düzelt
        updated_labels.append([category_id, new_x_center, new_y_center, new_width, new_height])

    # Görüntü ve etiketleri kaydet
    cv2.imwrite(output_image_path, augmented_image)
    save_yolo_labels(output_label_path, updated_labels)


# Veri setini işle
input_image_dir = "../drone_dataset_yolo/dataset_txt/images/train/"
input_label_dir = "../drone_dataset_yolo/dataset_txt/labels/train/"
output_image_dir = "../drone_dataset_yolo/dataset_txt/images/train_augmented/"
output_label_dir = "../drone_dataset_yolo/dataset_txt/labels/train_augmented/"

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

for file_name in os.listdir(input_image_dir):
    if file_name.endswith(".jpg"):
        image_path = os.path.join(input_image_dir, file_name)
        label_path = os.path.join(input_label_dir, file_name.replace(".jpg", ".txt"))
        output_image_path = os.path.join(output_image_dir, file_name)
        output_label_path = os.path.join(output_label_dir, file_name.replace(".jpg", ".txt"))

        if os.path.exists(label_path):  # Etiket dosyası yoksa işleme alma
            augment_image_and_labels(image_path, label_path, output_image_path, output_label_path)

print(" Veri artırma işlemi tamamlandı! Tüm etiketler YOLOv8 formatına uygun.")
