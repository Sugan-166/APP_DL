import tensorflow as tf
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

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

# Load Model Weights
def load_model_weights(model, weights_path):
    if os.path.isfile(weights_path):
        try:
            model.load_weights(weights_path)
        except Exception as e:
            print(f"Error loading weights: {e}")
    else:
        print(f"File not found: {weights_path}")
    return model

# Enhance Region Resolution with Color Overlay
def enhance_region_resolution_with_fade(model, region, color=(255, 105, 180)):
    # Resize to model input size
    region_resized = cv2.resize(region, (256, 256))
    region_norm = np.expand_dims(region_resized, axis=(0, -1)).astype('float32') / 255.0

    # Apply the model
    region_tensor = tf.convert_to_tensor(region_norm, dtype=tf.float32)
    high_res_region = model.predict(region_tensor)[0]
    high_res_region = (high_res_region * 255).astype(np.uint8)
    high_res_region_resized = cv2.resize(high_res_region, (region.shape[1], region.shape[0]))  # Resize back to original size

    # Convert high_res_region_resized to a 3-channel image
    high_res_region_rgb = cv2.cvtColor(high_res_region_resized, cv2.COLOR_GRAY2RGB)

    # Create a colored overlay with some transparency
    overlay = np.full_like(high_res_region_rgb, color, dtype=np.uint8)  # Create overlay with specified color
    
    # Blend the enhanced region with the color overlay
    blended_region = cv2.addWeighted(high_res_region_rgb, 0.7, overlay, 0.3, 0)  # Adjust weights for desired fade
    
    return blended_region

# Process Image and Enhance Detected Regions
def process_image_and_enhance_regions(model, image_path, output_path, color=(255, 105, 180), detect_high=True):
    # Load the original image in grayscale
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Detect regions based on intensity
    if detect_high:
        contours = detect_high_intensity_regions(original_image)
    else:
        contours = detect_low_intensity_regions(original_image)
    
    if not contours:
        return original_image, None
    
    # Create an enhanced image copy in RGB
    enhanced_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_end, y_end = x + w, y + h
        
        # Extract and enhance the detected region with a color overlay
        region = original_image[y:y_end, x:x_end]
        enhanced_region = enhance_region_resolution_with_fade(model, region, color)
        
        # Ensure sizes match before placing the enhanced region back into the image
        if enhanced_region.shape[:2] == (y_end - y, x_end - x):
            enhanced_image[y:y_end, x:x_end] = enhanced_region
        else:
            print(f"Size mismatch: {enhanced_region.shape} vs {(y_end - y, x_end - x)}")
    
    cv2.imwrite(output_path, enhanced_image)
    return original_image, enhanced_image

# Detect Low-Intensity Regions Using Simple Thresholding
def detect_low_intensity_regions(image, threshold=50):
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Detect High-Intensity Regions Using Simple Thresholding
def detect_high_intensity_regions(image, threshold=200):
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Draw Detected Regions on Image with Rectangles
def draw_detected_regions(image, contours, color=(255, 105, 180), thickness=2):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_end, y_end = x + w, y + h
        
        # Ensure coordinates are within image bounds
        x = max(x, 0)
        y = max(y, 0)
        x_end = min(x_end, image_rgb.shape[1])
        y_end = min(y_end, image_rgb.shape[0])
        
        # Draw rectangles
        cv2.rectangle(image_rgb, (x, y), (x_end, y_end), color, thickness)
    
    return image_rgb

# Convert OpenCV Image to PIL Image
def opencv_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Tkinter GUI
class ImageProcessingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Processing")
        self.geometry("1200x600")
        
        self.detect_intensity = tk.StringVar(value="low")  # Variable to store user choice for intensity detection

        # Radio buttons for choosing intensity detection type
        self.low_intensity_radio = tk.Radiobutton(self, text="Detect Low-Intensity Regions", variable=self.detect_intensity, value="low")
        self.low_intensity_radio.pack(pady=5)
        self.high_intensity_radio = tk.Radiobutton(self, text="Detect High-Intensity Regions", variable=self.detect_intensity, value="high")
        self.high_intensity_radio.pack(pady=5)

        self.upload_button = tk.Button(self, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=20)

        self.canvas_original = tk.Canvas(self, width=512, height=512)
        self.canvas_original.pack(side=tk.LEFT, padx=10)

        self.canvas_enhanced = tk.Canvas(self, width=512, height=512)
        self.canvas_enhanced.pack(side=tk.LEFT, padx=10)

        self.model_sr = create_super_resolution_model()
        self.model_sr = load_model_weights(self.model_sr, r'C:\Users\sugan\Desktop\Blurr regnition\output_images.h5')

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            # Determine the type of intensity detection based on user choice
            detect_high = self.detect_intensity.get() == "high"

            # Process the image and enhance detected regions
            original_image, enhanced_image = process_image_and_enhance_regions(self.model_sr, file_path, 'enhanced_image.png', detect_high=detect_high)
            
            if enhanced_image is None:
                # No regions detected
                messagebox.showinfo("No Regions Detected", "No regions were detected in the image.")
                return
            
            # Draw rectangles around detected regions
            if detect_high:
                contours = detect_high_intensity_regions(original_image)
            else:
                contours = detect_low_intensity_regions(original_image)
            
            original_image_with_regions = draw_detected_regions(original_image, contours, color=(255, 105, 180))
            cv2.imwrite('original_image_with_regions.png', original_image_with_regions)
            
            # Display images
            self.display_image(file_path, self.canvas_original, "Original Image")
            self.display_image('enhanced_image.png', self.canvas_enhanced, "Enhanced Image")
            
            # Open new window to show original image with detected regions
            self.show_processed_image('original_image_with_regions.png')

    def display_image(self, img_path, canvas, title):
        img = Image.open(img_path)
        img = img.resize((512, 512))  # Resize for display
        img = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        canvas.image = img
        
        # Update canvas title
        canvas_title = tk.Label(self, text=title)
        canvas_title.pack(side=tk.LEFT)

    def show_processed_image(self, img_path):
        processed_window = tk.Toplevel(self)
        processed_window.title("Original Image with Detected Regions")
        processed_window.geometry("600x600")
        
        canvas_processed = tk.Canvas(processed_window, width=512, height=512)
        canvas_processed.pack(pady=20)
        
        img = Image.open(img_path)
        img = img.resize((512, 512))  # Resize for display
        img = ImageTk.PhotoImage(img)
        canvas_processed.create_image(0, 0, anchor=tk.NW, image=img)
        canvas_processed.image = img

if __name__ == "__main__":
    app = ImageProcessingApp()
    app.mainloop()
