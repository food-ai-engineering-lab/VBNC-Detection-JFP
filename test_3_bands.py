import os
import spectral.io.envi as envi
from PIL import Image
import numpy as np
 
def process_folder(input_path, output_path, label):
    counter = 1
    for folder in os.listdir(input_path):
        folder_path = os.path.join(input_path, folder)
        if os.path.isdir(folder_path):
            hdr_path = None
            dat_path = None
            # Find the .hdr and .dat files in the folder
            for file in os.listdir(folder_path):
                if file.endswith('.hdr'):
                    hdr_path = os.path.join(folder_path, file)
                elif file.endswith('.dat'):
                    dat_path = os.path.join(folder_path, file)
            # Process the files if both are found
            if hdr_path and dat_path:
                img = envi.open(hdr_path, dat_path)
                # Adjust band indices based on your specific dataset
                band_red = img.read_band(24)
                band_green = img.read_band(96)
                band_blue = img.read_band(137)
                # Stack the bands to form an RGB image
                rgb_image = np.dstack((band_red, band_green, band_blue))
                # Normalize the bands to the range 0-255
                rgb_image_normalized = ((rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min()) * 255).astype('uint8')
                # Create and save the image
                img_png = Image.fromarray(rgb_image_normalized)
                img_png.save(os.path.join(output_path, f'{label}_{counter}.png'))
                counter += 1
 
# Specify the path to your data directory here
input_dir = '/mnt/data/cifs/HMI/0407-Ecoli-Live'
output_dir = './outputs/Normal'
output_label = 'Normal'

process_folder(input_dir, output_dir)
