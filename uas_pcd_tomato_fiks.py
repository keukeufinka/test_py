# uas_pcd_tomato_fiks_streamlit.py
import streamlit as st
import os
from zipfile import ZipFile
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

st.title("Tomato Leaf Disease Detection")

# -----------------------------------------
# Dataset extraction (hanya jika zip ada)
# -----------------------------------------
if os.path.exists("tomatoleaf.zip"):
    if not os.path.exists("tomatoleaf"):
        with ZipFile("tomatoleaf.zip", 'r') as zip_ref:
            zip_ref.extractall("tomatoleaf")
        st.success("Dataset berhasil diekstrak!")

# -----------------------------------------
# Upload gambar untuk prediksi
# -----------------------------------------
uploaded_file = st.file_uploader("Upload gambar daun tomat", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB").resize((224,224))
    input_data = np.expand_dims(np.array(img, dtype=np.float32)/255.0, axis=0)

    # -----------------------------------------
    # Load model TFLite
    # -----------------------------------------
    if not os.path.exists("model.tflite"):
        st.error("File model.tflite tidak ditemukan!")
    else:
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

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

        st.image(img, caption=f"Prediksi: {class_names[pred_class]}")
        st.write(f"Prediksi kelas: {class_names[pred_class]}")
