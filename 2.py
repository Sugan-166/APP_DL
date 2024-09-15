import h5py
import numpy as np
from PIL import Image
import os

def folder_to_h5_with_folders(root_folder_path, h5_file):
    # Create a new HDF5 file
    with h5py.File(h5_file, 'w') as hf:
        # Walk through the root folder and its subdirectories
        for dirpath, _, filenames in os.walk(root_folder_path):
            # Skip empty directories
            if not filenames:
                continue
            
            # Create a group in HDF5 for the current directory
            rel_path = os.path.relpath(dirpath, root_folder_path)
            group_path = '/' + rel_path.replace(os.sep, '/')
            group = hf.require_group(group_path)

            # Process each PNG file in the directory
            for filename in filenames:
                if filename.endswith('.png'):
                    file_path = os.path.join(dirpath, filename)
                    
                    # Open the PNG file
                    image = Image.open(file_path)
                    
                    # Convert image to numpy array
                    image_array = np.array(image)
                    
                    # Create a dataset in the current group
                    dataset_name = os.path.splitext(filename)[0]  # Use filename without extension as dataset name
                    group.create_dataset(dataset_name, data=image_array)

# Corrected Example usage
folder_to_h5_with_folders(r'C:\Users\sugan\Desktop\h5', 'output_images.h5')
