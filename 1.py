import tensorflow as tf
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Define Super-Resolution Model
def create_super_resolution_model(input_shape=(256, 256, 1)):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Convolutional layers
    conv1 = tf.keras.layers.Conv2D(64, (9, 9), activation='relu', padding='same')(inputs)
    conv2 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same')(conv1)
    conv3 = tf.keras.layers.Conv2D(1, (5, 5), padding='same')(conv2)
    
    # Upsampling layer
    upsample = tf.keras.layers.UpSampling2D(size=(2, 2))(conv3)
    
    model = tf.keras.Model(inputs=inputs, outputs=upsample)
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# Load Super-Resolution Model Weights
def load_model_weights(model, weights_path):
    try:
        model.load_weights(weights_path)
    except Exception as e:
        print(f"Error loading weights: {e}")
    return model

# Enhance Image Resolution
def enhance_image_resolution(model, image_path, output_path):
    # Load and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256))  # Resize to model input size
    image_norm = np.expand_dims(image, axis=(0, -1)).astype('float32') / 255.0

    # Apply the model
    image_tensor = tf.convert_to_tensor(image_norm, dtype=tf.float32)
    high_res_image = model.predict(image_tensor)[0]
    high_res_image = (high_res_image * 255).astype(np.uint8)
    high_res_image = cv2.resize(high_res_image, (512, 512))  # Resize to desired output size

    # Save the high-resolution image
    cv2.imwrite(output_path, high_res_image)
    return high_res_image

# Detect Regions (Example: High-Intensity Regions)
def detect_regions(image, threshold=200):
    # Convert image to binary based on intensity threshold
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours (regions) in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

# Draw Detected Regions on Image
def draw_detected_regions(image, contours):
    # Convert grayscale image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Draw contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return image_rgb

# Convert OpenCV Image to PIL Image
def opencv_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Tkinter GUI
class SuperResolutionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Radar Image Super-Resolution")
        self.geometry("1200x600")
        
        self.upload_button = tk.Button(self, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=20)

        # Canvases for original and high-resolution images
        self.canvas_original = tk.Canvas(self, width=512, height=512)
        self.canvas_original.pack(side=tk.LEFT, padx=10)

        self.canvas_high_res = tk.Canvas(self, width=512, height=512)
        self.canvas_high_res.pack(side=tk.LEFT, padx=10)

        # Load the model
        self.model = create_super_resolution_model()
        self.model = load_model_weights(self.model, r'C:\Users\sugan\Desktop\Blurr regnition\output_images.h5')  # Update with actual path

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            # Enhance resolution
            output_path = 'high_res_image.png'
            high_res_image = enhance_image_resolution(self.model, file_path, output_path)
            
            # Detect regions
            contours = detect_regions(high_res_image)
            high_res_image_with_regions = draw_detected_regions(high_res_image, contours)
            
            # Save the image with detected regions
            cv2.imwrite('high_res_image_with_regions.png', high_res_image_with_regions)
            
            # Display images
            self.display_image(file_path, self.canvas_original, "Original Image")
            self.display_image('high_res_image_with_regions.png', self.canvas_high_res, "Detected Regions")

    def display_image(self, img_path, canvas, title):
        img = Image.open(img_path)
        img = img.resize((512, 512))  # Resize for display
        img = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        canvas.image = img  # Keep a reference to avoid garbage collection
        
        # Update canvas title
        canvas_title = tk.Label(self, text=title)
        canvas_title.pack(side=tk.LEFT)

if __name__ == "__main__":
    app = SuperResolutionApp()
    app.mainloop()
