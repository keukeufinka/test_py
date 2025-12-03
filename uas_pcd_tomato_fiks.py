# -*- coding: utf-8 -*-
"""
UAS_PCD_TOMATO_FIKS.py
Versi Python yang bisa dijalankan di MacBook / Terminal
"""

import os
import shutil
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator, image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report

# =========================
# 1. Setup Kaggle folder
# =========================
kaggle_json = "kaggle.json"  # letakkan kaggle.json di folder project
kaggle_folder = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_folder, exist_ok=True)
shutil.copy(kaggle_json, os.path.join(kaggle_folder, "kaggle.json"))
os.chmod(os.path.join(kaggle_folder, "kaggle.json"), 0o600)
print("Kaggle JSON siap!")

# =========================
# 2. Download dan ekstrak dataset
# =========================
# Pastikan kaggle CLI sudah terinstall
os.system("kaggle datasets download -d kaustubhb999/tomatoleaf -p ./")
if not os.path.exists("tomatoleaf"):
    os.makedirs("tomatoleaf")
with zipfile.ZipFile("tomatoleaf.zip", 'r') as zip_ref:
    zip_ref.extractall("tomatoleaf")
print("Dataset siap di folder 'tomatoleaf'")

# =========================
# 3. Explore Data
# =========================
train_dir = "tomatoleaf/tomato/train"
val_dir = "tomatoleaf/tomato/val"

print("Isi folder tomato:", os.listdir("tomatoleaf/tomato"))
print("Train classes:", os.listdir(train_dir))
print("Val classes:", os.listdir(val_dir))

# Hitung jumlah gambar per kelas
train_counts = {folder: len(os.listdir(os.path.join(train_dir, folder))) for folder in os.listdir(train_dir)}
val_counts = {folder: len(os.listdir(os.path.join(val_dir, folder))) for folder in os.listdir(val_dir)}

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.bar(train_counts.keys(), train_counts.values(), color='skyblue')
plt.title('Distribusi Gambar per Kelas (Train)')
plt.xticks(rotation=90)
plt.ylabel('Jumlah Gambar')

plt.subplot(1,2,2)
plt.bar(val_counts.keys(), val_counts.values(), color='lightgreen')
plt.title('Distribusi Gambar per Kelas (Validation)')
plt.xticks(rotation=90)
plt.ylabel('Jumlah Gambar')
plt.tight_layout()
plt.show()

# Tampilkan satu gambar contoh dari tiap kelas
classes = os.listdir(train_dir)
plt.figure(figsize=(15,8))
for i, folder in enumerate(classes):
    img_path = os.path.join(train_dir, folder, os.listdir(os.path.join(train_dir, folder))[0])
    img = load_img(img_path, target_size=(150,150))
    plt.subplot(2,5,i+1)
    plt.imshow(img)
    plt.title(folder.split("___")[-1])
    plt.axis('off')
plt.suptitle("Contoh Daun Tomat: Sehat & 9 Penyakit", fontsize=16)
plt.show()

# =========================
# 4. Preprocessing & Edge Detection (Opsional)
# =========================
sample_path = os.path.join(train_dir, "Tomato___Late_blight")
sample_image = os.path.join(sample_path, os.listdir(sample_path)[0])
img = cv2.imread(sample_image)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
edges = cv2.Canny(blur, 100, 200)

plt.figure(figsize=(12,4))
plt.subplot(1,4,1); plt.imshow(img_rgb); plt.title('RGB Asli'); plt.axis('off')
plt.subplot(1,4,2); plt.imshow(gray, cmap='gray'); plt.title('Grayscale'); plt.axis('off')
plt.subplot(1,4,3); plt.imshow(blur, cmap='gray'); plt.title('Noise Reduction'); plt.axis('off')
plt.subplot(1,4,4); plt.imshow(edges, cmap='gray'); plt.title('Edge Detection'); plt.axis('off')
plt.suptitle("Tahapan Pengolahan Citra Daun Tomat", fontsize=16)
plt.show()

# =========================
# 5. Data Preparation
# =========================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# =========================
# 6. Modeling
# =========================
base_model = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
base_model.summary()

# =========================
# 7. Training
# =========================
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
callbacks_list = [early_stop, reduce_lr, checkpoint]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=callbacks_list
)

best_model = load_model('best_model.keras')
val_loss, val_accuracy = best_model.evaluate(val_generator)
print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# =========================
# 8. Confusion Matrix & Classification Report
# =========================
predictions = best_model.predict(val_generator)
pred_classes = np.argmax(predictions, axis=1)
y_true = val_generator.classes

cm = confusion_matrix(y_true, pred_classes)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=val_generator.class_indices.keys(),
            yticklabels=val_generator.class_indices.keys())
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_true, pred_classes, target_names=val_generator.class_indices.keys()))

# =========================
# 9. Uji Gambar Tunggal
# =========================
img_path = "tomato6.jpeg"  # path lokal
img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

pred = best_model.predict(img_array)
pred_label = list(val_generator.class_indices.keys())[np.argmax(pred)]
confidence = np.max(pred)

plt.imshow(img)
plt.title(f"Predicted: {pred_label}\nConfidence: {confidence:.2f}")
plt.axis('off')
plt.show()

print("Prediksi kelas:", pred_label)

# =========================
# 10. Save Model ke TFLite
# =========================
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
print("Model TFLite tersimpan di 'model.tflite'")

# =========================
# 11. Uji Model TFLite
# =========================
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
height, width = input_shape[1], input_shape[2]

img = Image.open(img_path).convert("RGB").resize((width, height))
input_data = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
pred_class = np.argmax(output_data[0])

class_names = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

plt.imshow(img)
plt.title(f"Prediksi: {class_names[pred_class]}")
plt.axis('off')
plt.show()
print("Prediksi kelas (TFLite):", class_names[pred_class])
