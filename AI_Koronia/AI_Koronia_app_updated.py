import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import altair as alt
import pandas as pd
import pytesseract
import re
import time
import matplotlib.pyplot as plt

# Set Tesseract executable path (update if necessary)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

###############################
# Utility Functions
###############################

def average_color(image):
    """Compute the average color of the image, ignoring near-black/near-white pixels."""
    arr = np.array(image)
    mask_black = (arr[:,:,0] < 30) & (arr[:,:,1] < 30) & (arr[:,:,2] < 30)
    mask_white = (arr[:,:,0] > 225) & (arr[:,:,1] > 225) & (arr[:,:,2] > 225)
    mask = ~(mask_black | mask_white)
    if np.sum(mask) == 0:
        return np.mean(arr.reshape(-1, 3), axis=0)
    else:
        return np.mean(arr[mask], axis=0)

def determine_category(avg):
    """
    Determine category based on average color:
      - "moderate" if green is dominant,
      - "good" if blue is dominant and red is relatively low,
      - "bad" if red is dominant or if blue is dominant but red is at least 80% of blue (purple/pink).
    """
    r, g, b = avg
    if g >= r and g >= b:
        return "moderate"
    elif b >= r and b >= g:
        if r >= 0.8 * b:
            return "bad"
        else:
            return "good"
    else:
        return "bad"

def extract_date_from_frame(frame, crop_box=None):
    """
    Extracts a date in the format YYYY-MM-DD from the top-right corner of a frame using OCR.
    crop_box: tuple (left, upper, right, lower). Adjust as needed.
    Returns the extracted date string if found; otherwise, returns None.
    """
    width, height = frame.size
    if crop_box is None:
        crop_box = (width - 150, 0, width, 60)
    date_region = frame.crop(crop_box).convert("L")
    # Binarize the image (adjust threshold as needed)
    date_region = date_region.point(lambda x: 0 if x < 100 else 255, '1')
    custom_config = r'--oem 3 --psm 7'
    text = pytesseract.image_to_string(date_region, config=custom_config)
    text = text.strip(" \n\r\t,«.")
    matches = re.findall(r'(\d{4}-\d{2}-\d{2})', text)
    if matches:
        return matches[0]
    else:
        return None

@st.cache_data
def extract_frames(uploaded_file):
    """Extracts frames from a GIF, saves them in /tmp/frames_jpg, and returns frames, saved_counts, and total frame count."""
    gif = Image.open(uploaded_file)
    frame_number = 0
    frames = []
    saved_counts = {"good": 0, "moderate": 0, "bad": 0}
    base_folder = os.path.join("/tmp", "frames_jpg")
    # Create subfolders in a writable directory
    for cat in ["good", "moderate", "bad"]:
        os.makedirs(os.path.join(base_folder, cat), exist_ok=True)
    while True:
        try:
            gif.seek(frame_number)
            frame = gif.convert("RGB")
            frames.append(frame)
            avg = average_color(frame)
            category = determine_category(avg)
            frame_dir = os.path.join(base_folder, category)
            os.makedirs(frame_dir, exist_ok=True)
            frame_path = os.path.join(frame_dir, f"frame_{frame_number}.jpg")
            frame.save(frame_path, "JPEG")
            saved_counts[category] += 1
            frame_number += 1
        except EOFError:
            break
    return frames, saved_counts, frame_number

@st.cache_resource
def load_model():
    """Load the trained CNN model."""
    model = tf.keras.models.load_model("AI_Koronia_Model_6.keras", compile=False)
    return model

def train_model(num_epochs):
    """Train the CNN model on your dataset and return the model, history, and evaluation."""
    batch_size = 32
    img_height = 128
    img_width = 128
    train_dir = r"./data/training"  # Use relative paths for deployment
    test_dir  = r"./data/testing"
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=True
    )
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=False
    )
    
    X_train = np.concatenate([images.numpy() for images, labels in train_ds], axis=0)
    Y_train = np.concatenate([labels.numpy() for images, labels in train_ds], axis=0)
    X_test = np.concatenate([images.numpy() for images, labels in test_ds], axis=0)
    Y_test = np.concatenate([labels.numpy() for images, labels in test_ds], axis=0)
    
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (4,4), activation='relu', input_shape=(img_height, img_width, 3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Conv2D(32, (4,4), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    st.write("Starting training for", num_epochs, "epochs...")
    start_time = time.time()
    history = model.fit(X_train, Y_train, batch_size=16, epochs=num_epochs, verbose=1, validation_data=(X_test, Y_test))
    elapsed = time.time() - start_time
    st.write(f"Training completed in {elapsed:.2f} seconds.")
    
    evaluation = model.evaluate(X_test, Y_test)
    st.write("Test Accuracy:", evaluation[1])
    model.save("AI_Koronia_Model_6.keras")
    return model, history, evaluation

###############################
# Sidebar: Mode Selection
###############################
mode = st.sidebar.radio("Select Mode", ["Extract Frames for Training", "Train Model", "Test Model"])

###############################
# Mode 1: Extract Frames for Training
###############################
if mode == "Extract Frames for Training":
    st.title("Extract GIF Frames for Training")
    st.write("Upload a GIF file. Frames will be extracted and grouped by dominant color into subfolders under '/tmp/frames_jpg'.")
    uploaded_file = st.file_uploader("Upload a GIF file", type=["gif"], key="gif_train")
    
    if uploaded_file is not None:
        frames, saved_counts, frame_number = extract_frames(uploaded_file)
        base_folder = os.path.join("/tmp", "frames_jpg")
        st.success(f"Extracted {frame_number} frames and saved them under '{base_folder}': {saved_counts}")
        for cat in ["good", "moderate", "bad"]:
            cat_folder = os.path.join(base_folder, cat)
            if os.path.exists(cat_folder):
                files = os.listdir(cat_folder)
                if files:
                    images = [Image.open(os.path.join(cat_folder, f)) for f in files[:3]]
                    st.image(images, caption=[f"Examples from {cat}"] * len(images), width=150)

###############################
# Mode 2: Train Model
###############################
elif mode == "Train Model":
    st.title("Train the CNN Model")
    st.write("This process trains the CNN model on your dataset and saves the model as 'AI_Koronia_Model_6.keras'.")
    num_epochs = st.number_input("Enter number of epochs", min_value=1, max_value=100, value=5, step=1, key="epochs_train")
    if st.button("Start Training", key="train_button"):
        with st.spinner("Training model..."):
            model, history, evaluation = train_model(num_epochs)
            st.success("Training complete!")
            st.write(f"Test Accuracy: {evaluation[1]:.4f}")
            fig, ax = plt.subplots()
            ax.plot(history.history['loss'], label='Train Loss')
            ax.plot(history.history['val_loss'], label='Validation Loss')
            ax.plot(history.history['accuracy'], label='Train Accuracy')
            ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss / Accuracy")
            ax.legend()
            st.pyplot(fig)

###############################
# Mode 3: Test Model (with Monthly Bar Chart & Percentage)
###############################
elif mode == "Test Model":
    st.title("Test Computer Vision Model with Monthly Bar Chart")
    st.write("Upload a GIF file for prediction. The app will classify each frame, extract the date (YYYY-MM-DD) from the top-right, and produce a bar chart showing the monthly distribution and percentage of frames predicted as good, moderate, or bad.")
    uploaded_file = st.file_uploader("Upload an image", type=["gif", "jpg", "jpeg", "png"], key="gif_test")
    
    if uploaded_file is not None:
        if uploaded_file.type == "image/gif":
            gif = Image.open(uploaded_file)
            frames = []
            dates = []
            frame_number = 0
            last_valid_date = None
            while True:
                try:
                    gif.seek(frame_number)
                    frame = gif.convert("RGB")
                    frames.append(frame)
                    date_str = extract_date_from_frame(frame)
                    if date_str is None and last_valid_date is not None:
                        date_str = last_valid_date
                    elif date_str is not None:
                        last_valid_date = date_str
                    dates.append(date_str)
                    frame_number += 1
                except EOFError:
                    break
            
            st.write(f"Extracted {len(frames)} frames from the GIF.")
            st.write("Extracted Dates:", dates)
            
            model = load_model()
            class_names = ["bad", "good", "moderate"]
            predictions = []
            for fr in frames:
                img_size = (128, 128)
                image_resized = ImageOps.fit(fr, img_size, Image.Resampling.LANCZOS)
                img_array = np.array(image_resized).astype("float32") / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                pred = model.predict(img_array)
                predicted_class = np.argmax(pred, axis=1)[0]
                predictions.append(class_names[predicted_class])
            
            df = pd.DataFrame({
                "FrameIndex": list(range(len(frames))),
                "PredictedState": predictions,
                "Date": dates
            })
            
            def convert_date(x):
                try:
                    return pd.to_datetime(x, format="%Y-%m-%d")
                except Exception:
                    return pd.NaT
            df["DateTime"] = df["Date"].apply(lambda x: convert_date(x) if x is not None else pd.NaT)
            df["Month"] = df["DateTime"].dt.to_period("M").astype(str)
            
            df_valid = df.dropna(subset=["DateTime"])
            if df_valid.empty:
                st.warning("No valid dates extracted from frames. Cannot create a monthly bar chart.")
            else:
                counts = df_valid.groupby(["Month", "PredictedState"]).size().reset_index(name="Count")
                counts["Total"] = counts.groupby("Month")["Count"].transform("sum")
                counts["Percentage"] = counts["Count"] / counts["Total"] * 100
                
                st.write("## Monthly Distribution of Predicted States (Counts & Percentages)")
                chart = alt.Chart(counts).mark_bar().encode(
                    x=alt.X("Month:N", title="Month"),
                    y=alt.Y("Percentage:Q", title="Percentage of Frames"),
                    color=alt.Color("PredictedState:N", title="Predicted State",
                                     scale=alt.Scale(domain=["bad", "moderate", "good"],
                                                     range=["red", "green", "blue"])),
                    tooltip=["Month", "PredictedState", "Count", alt.Tooltip("Percentage:Q", format=".1f")]
                ).properties(width=600)
                st.altair_chart(chart, use_container_width=True)
                
                st.write("### Detailed Frame Data")
                st.dataframe(df_valid[["FrameIndex", "Date", "PredictedState"]])
            
            st.write("## View a Specific Frame")
            frame_index = st.number_input("Select frame number", min_value=0, max_value=len(frames)-1, value=0, step=1, key="frame_view")
            selected_frame = frames[frame_index]
            st.image(selected_frame, caption=f"Frame {frame_index}", use_column_width=True)
            
            image_resized = ImageOps.fit(selected_frame, (128, 128), Image.Resampling.LANCZOS)
            img_array = np.array(image_resized).astype("float32") / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_label = class_names[predicted_class]
            st.write(f"**Prediction for Frame {frame_index}:** {predicted_label}")
        
        else:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write("Classifying...")
            
            model = load_model()
            img_size = (128, 128)
            image_resized = ImageOps.fit(image, img_size, Image.Resampling.LANCZOS)
            img_array = np.array(image_resized).astype("float32") / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            class_names = ["bad", "good", "moderate"]
            predicted_label = class_names[predicted_class]
            st.write(f"**Prediction:** {predicted_label}")
            st.write("**Prediction Probabilities:**", prediction)
