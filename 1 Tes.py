import tensorflow as tf
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

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
    image_resized = cv2.resize(image, (256, 256))  # Resize to model input size
    image_norm = np.expand_dims(image_resized, axis=(0, -1)).astype('float32') / 255.0

    # Apply the model
    image_tensor = tf.convert_to_tensor(image_norm, dtype=tf.float32)
    high_res_image = model.predict(image_tensor)[0]
    high_res_image = (high_res_image * 255).astype(np.uint8)
    high_res_image = cv2.resize(high_res_image, (512, 512))  # Resize to desired output size

    # Save the high-resolution image
    cv2.imwrite(output_path, high_res_image)

    # Resize original image to match high-resolution output
    original_image_resized = cv2.resize(image_resized, (512, 512))
    
    return high_res_image, original_image_resized

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

# Plot regression-type graph with actual and predicted images
def plot_regression(actual_image, predicted_image):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Actual Image Plot
    axs[0].imshow(actual_image, cmap='gray')
    axs[0].set_title('Actual Image')
    axs[0].axis('off')

    # Predicted Image Plot
    axs[1].imshow(predicted_image, cmap='gray')
    axs[1].set_title('Predicted Image')
    axs[1].axis('off')
    
    # Create a Tkinter window for the graph
    graph_window = tk.Toplevel()
    graph_window.title("Predicted Regions")

    canvas = FigureCanvasTkAgg(fig, master=graph_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Plot line graph for image pixel values
def plot_line_graph(actual_image, predicted_image):
    # Ensure images are of the same dimensions
    if actual_image.shape != predicted_image.shape:
        raise ValueError("Actual and predicted images must have the same dimensions for comparison.")
    
    # Flatten images and create x-axis values
    actual_flat = actual_image.flatten()
    predicted_flat = predicted_image.flatten()
    x_values = np.arange(len(actual_flat))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_values, actual_flat, label='Actual Image', color='blue', linestyle='-', linewidth=1.5)
    ax.plot(x_values, predicted_flat, label='Predicted Image', color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Pixel Index')
    ax.set_ylabel('Pixel Value')
    ax.set_title('Pixel Value Comparison')
    ax.legend()

    # Create a Tkinter window for the graph
    graph_window = tk.Toplevel()
    graph_window.title("Line Graph")

    canvas = FigureCanvasTkAgg(fig, master=graph_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Plot model parameters graph
def plot_graph(parameters):
    # Create a new Tkinter window for the graph
    graph_window = tk.Toplevel()
    graph_window.title("Graph Parameters")

    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract parameter names and values
    names, values = zip(*parameters)

    # Create a bar plot
    ax.bar(names, values, color='skyblue')
    ax.set_xlabel('Parameter')
    ax.set_ylabel('Value')
    ax.set_title('Model Graph Parameters')

    # Add the plot to the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=graph_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Calculate Accuracy and Prediction Rate using PSNR and SSIM
def calculate_metrics(model, image_path, ground_truth_path):
    # Load and preprocess the images
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize images to match the model input size
    image_resized = cv2.resize(image, (256, 256))
    ground_truth_resized = cv2.resize(ground_truth, (256, 256))
    
    # Normalize images
    image_norm = np.expand_dims(image_resized, axis=(0, -1)).astype('float32') / 255.0
    ground_truth_norm = np.expand_dims(ground_truth_resized, axis=(0, -1)).astype('float32') / 255.0
    
    # Apply the super-resolution model
    image_tensor = tf.convert_to_tensor(image_norm, dtype=tf.float32)
    predicted_image = model.predict(image_tensor)[0]
    predicted_image = (predicted_image * 255).astype(np.uint8)
    
    # Resize predicted image to match ground truth
    predicted_image_resized = cv2.resize(predicted_image, (256, 256))
    
    # Calculate metrics
    psnr_value = psnr(ground_truth_resized, predicted_image_resized)
    ssim_value = ssim(ground_truth_resized, predicted_image_resized, data_range=predicted_image_resized.max() - predicted_image_resized.min())
    
    # Convert PSNR and SSIM to a scale of 0 to 1000 for comparison
    psnr_score = (psnr_value / 100) * 1000  # Example: scaling PSNR to 0-1000 range
    ssim_score = ssim_value * 1000  # Example: scaling SSIM to 0-1000 range
    
    return psnr_score, ssim_score

# Tkinter GUI
class SuperResolutionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Radar Image Super-Resolution")
        self.geometry("1600x800")
        
        self.upload_button = tk.Button(self, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=20)

        # Canvases for original and high-resolution images
        self.canvas_original = tk.Canvas(self, width=512, height=512)
        self.canvas_original.pack(side=tk.LEFT, padx=10)

        self.canvas_high_res = tk.Canvas(self, width=512, height=512)
        self.canvas_high_res.pack(side=tk.LEFT, padx=10)

        # Labels for metrics
        self.label_predicted_rate = tk.Label(self, text="Predicted Rate: N/A")
        self.label_predicted_rate.pack(pady=10)

        self.label_accuracy_score = tk.Label(self, text="Accuracy Score: N/A")
        self.label_accuracy_score.pack(pady=10)

        self.label_graph_parameters = tk.Button(self, text="Show Graph Parameters", command=self.show_graph_parameters)
        self.label_graph_parameters.pack(pady=10)

        # Load the super-resolution model
        self.model = create_super_resolution_model()
        self.model = load_model_weights(self.model, r'C:\Users\sugan\Desktop\Blurr regnition\output_images.h5')  # Update with actual path

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            # Enhance resolution using the selected image
            output_path = 'high_res_image.png'
            high_res_image, original_image = enhance_image_resolution(self.model, file_path, output_path)
            
            # For demonstration, we'll use the same image as the ground truth
            ground_truth_path = file_path
            
            # Detect regions
            contours = detect_regions(high_res_image)
            high_res_image_with_regions = draw_detected_regions(high_res_image, contours)
            
            # Save the image with detected regions
            cv2.imwrite('high_res_image_with_regions.png', high_res_image_with_regions)
            
            # Calculate metrics
            psnr_score, ssim_score = calculate_metrics(self.model, file_path, ground_truth_path)

            # Update metrics labels
            self.label_predicted_rate.config(text=f"PSNR Score: {psnr_score:.2f}")
            self.label_accuracy_score.config(text=f"SSIM Score: {ssim_score:.2f}")
            
            # Display images
            self.display_image(file_path, self.canvas_original, "Original Image")
            self.display_image('high_res_image.png', self.canvas_high_res, "High-Resolution Image")

            # Plot regression graph
            plot_regression(original_image, high_res_image)
            # Plot line graph
            plot_line_graph(original_image, high_res_image)

            # Open a separate window to display the image with detected regions
            self.show_detected_regions('high_res_image_with_regions.png')

    def get_graph_parameters(self):
        # Placeholder for actual graph parameters
        return [('Parameter 1', 0.5), ('Parameter 2', 0.8), ('Parameter 3', 0.9)]  # Replace with actual logic

    def show_graph_parameters(self):
        parameters = self.get_graph_parameters()
        plot_graph(parameters)

    def display_image(self, img_path, canvas, title):
        img = Image.open(img_path)
        img = img.resize((512, 512))  # Resize for display
        img = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        canvas.image = img  # Keep a reference to avoid garbage collection
        
        # Update canvas title
        canvas_title = tk.Label(self, text=title)
        canvas_title.pack(side=tk.LEFT)

    def show_detected_regions(self, img_path):
        # Create a new window for displaying the image with detected regions
        detected_regions_window = tk.Toplevel(self)
        detected_regions_window.title("Detected Regions")

        canvas = tk.Canvas(detected_regions_window, width=512, height=512)
        canvas.pack()

        img = Image.open(img_path)
        img = img.resize((512, 512))  # Resize for display
        img = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        canvas.image = img  # Keep a reference to avoid garbage collection

if __name__ == "__main__":
    app = SuperResolutionApp()
    app.mainloop()
