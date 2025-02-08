import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# -----------------------------
# 1. Load the saved model
# -----------------------------
# Use a raw string (r"...") to avoid issues with backslashes in Windows paths.
  model_path = r"\AI_Koronia_Model.keras"
# model = tf.keras.models.load_model("./AI_Koronia_Model.keras", compile=False)


try:
    model = tf.keras.models.load_model(model_path)
   # model = tf.keras.models.load_model("./AI_Koronia_Model.keras", compile=False)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model. Please check the path.\n{e}")
    st.stop()  # Stop the app if the model can't be loaded

# -----------------------------
# 2. Streamlit App Title and Description
# -----------------------------
st.title("Image Classification App")
st.write("Upload an image and the model will predict its class.")

# -----------------------------
# 3. Image Upload Widget
# -----------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # -----------------------------
    # 4. Preprocess the Image
    # -----------------------------
    # Update this if your model uses a different input size.
    img_size = (128, 128)
    
    # Resize the image while keeping the aspect ratio intact.
    #image = ImageOps.fit(image, img_size, Image.ANTIALIAS)
    image = ImageOps.fit(image, img_size, Image.Resampling.LANCZOS)

    # Convert the image to a NumPy array
    img_array = np.array(image)
    
    # Normalize the image (if your training images were normalized)
    img_array = img_array.astype("float32") / 255.0
    
    # Expand dimensions to create a batch of size 1
    img_array = np.expand_dims(img_array, axis=0)
    
    # -----------------------------
    # 5. Make a Prediction
    # -----------------------------
    prediction = model.predict(img_array)
    
    # Get the predicted class (assuming the model outputs probabilities for each class)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Define a mapping from class indices to class labels (adjust based on your training setup)
    class_names = ["bad", "good", "moderate"]
    predicted_label = class_names[predicted_class]
    
    st.write(f"**Prediction:** {predicted_label}")
    st.write("**Prediction Probabilities:**", prediction)
