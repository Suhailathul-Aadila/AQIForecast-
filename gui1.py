import customtkinter as ctk
from tkinter import messagebox
from PIL import Image
import pandas as pd
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Setup GUI appearance
ctk.set_appearance_mode('light')
ctk.set_default_color_theme('blue')

# Load model and scalers
custom_objects = {'mse': MeanSquaredError()}
model1 = load_model('my_model.keras', custom_objects=custom_objects)
scaler1 = pickle.load(open('scaler1.pkl', 'rb'))
scaler2 = pickle.load(open('scaler2.pkl', 'rb'))

# Initialize main window
app = ctk.CTk()
app.title("AQI FORECASTING")
app.geometry("900x600")
app.resizable(False, False)

# Set app icon if it exists
if os.path.exists('earth.ico'):
    app.iconbitmap('earth.ico')
else:
    print("Warning: Icon file not found.")

# Left Frame (Image)
frame1 = ctk.CTkFrame(app, width=465, height=600, corner_radius=0)
frame1.place(x=0, y=0)

frame1_inner = ctk.CTkFrame(frame1, width=415, height=470, corner_radius=0)
frame1_inner.place(x=25, y=62)

# Load and display image
image_path = "images.jpg"  # Make sure this exists
if os.path.exists(image_path):
    loaded_img = Image.open(image_path)
    img = ctk.CTkImage(dark_image=loaded_img, size=(380, 380))
    icon_label = ctk.CTkLabel(frame1_inner, image=img, text="")
    icon_label.place(x=10, y=45)
else:
    print("Warning: Image file not found.")

# Right Frame (Form)
frame2 = ctk.CTkFrame(app, width=400, height=564, corner_radius=20)
frame2.place(x=480, y=20)

frame2_inner = ctk.CTkFrame(frame2, width=360, height=534, corner_radius=20)
frame2_inner.place(x=20, y=15)

label1 = ctk.CTkLabel(frame2_inner, text="AQI FORECASTING", font=ctk.CTkFont(size=18, weight="bold"))
label1.place(x=80, y=30)

# Entry Fields
fields = ["PM2.5", "PM10", "CO", "NOx", "NO", "NO2", "SO2"]
entries = {}
y_position = 100

for field in fields:
    label = ctk.CTkLabel(frame2_inner, text=f"{field}:", font=ctk.CTkFont(size=14))
    label.place(x=38, y=y_position)
    entry = ctk.CTkEntry(frame2_inner, width=140, height=38, corner_radius=20)
    entry.place(x=120, y=y_position)
    entries[field] = entry
    y_position += 50

# Predict Function
def predict():
    try:
        data = {field: float(entries[field].get()) for field in fields}
        df = pd.DataFrame([data])

        # Scale features
        scaled_data = scaler1.transform(df)

        # Prepare for LSTM input: shape (1, 1, n_features)
        model_input = np.expand_dims(scaled_data, axis=1)

        # Predict
        prediction = model1.predict(model_input)

        # prediction.shape is (1, 1) or (1,) depending on model structure.
        # Inverse scale the predicted AQI
        if prediction.ndim == 3:
            prediction = prediction[:, -1, :]  # From (1, 1, 1) to (1, 1)
        output = scaler2.inverse_transform(prediction)[0][0]
        output = int(output)

        severity = (
            "Good" if output <= 50 else
            "Satisfactory" if output <= 100 else
            "Moderate" if output <= 200 else
            "Poor" if output <= 300 else
            "Very Poor" if output <= 400 else
            "Severe"
        )

        messagebox.showinfo("Next Day AQI", f"AQI: {output}\nSeverity: {severity}")

    except Exception as e:
        print(e)
        messagebox.showerror("Error", f"Invalid input: {e}")



# Refresh Function
def refresh():
    for entry in entries.values():
        entry.delete(0, ctk.END)

# Buttons
predict_btn = ctk.CTkButton(frame2_inner, text="Predict AQI", command=predict, width=120, corner_radius=12)
predict_btn.place(x=50, y=480)

clear_btn = ctk.CTkButton(frame2_inner, text="Clear", command=refresh, width=120, corner_radius=12)
clear_btn.place(x=190, y=480)

# Start the GUI
app.mainloop()
