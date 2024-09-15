import cv2
import numpy as np
import os
from pathlib import Path

def process_image(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    
    # Apply Gaussian Blur to reduce noise
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Apply Histogram Equalization to improve contrast
    equalized_img = cv2.equalizeHist(blurred_img)
    
    return equalized_img

def process_images_in_folder(input_folder, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # Process the image
            processed_img = process_image(input_path)
            
            if processed_img is not None:
                # Save the processed image
                cv2.imwrite(output_path, processed_img)
                print(f"Processed and saved: {output_path}")

# Define input and output folders
input_folder = r'C:\Users\sugan\Desktop\low level enhancement\h5 1'
output_folder = r'C:\Users\sugan\Desktop\low level enhancement\denoised h5'

# Process images in the folder
process_images_in_folder(input_folder, output_folder)
