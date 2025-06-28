import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Set paths
data_dir = r"C:\GOAT\Data"  # Replace with the dataset path
output_dir = r"C:\GOAT\Processed_Output"
os.makedirs(output_dir, exist_ok=True)

# Image preprocessing function
def preprocess_image(image_path, target_size=(224, 224)):
    # Read the image
    image = cv2.imread(image_path)
    # Resize the image
    image = cv2.resize(image, target_size)
    # Normalize the image
    image = image / 255.0
    return image

# Preprocess and save images
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(root, file)
            processed_image = preprocess_image(file_path)
            output_path = os.path.join(output_dir, file)
            # Save the preprocessed image
            cv2.imwrite(output_path, (processed_image * 255).astype(np.uint8))

print("Image preprocessing complete. Preprocessed images saved.")
