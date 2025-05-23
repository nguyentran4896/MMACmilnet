import model
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import zoom


def main():
    # Parameters
    img_size = 800
    patch_size = 200
    stride = 100
    
    num_patches = (img_size - patch_size) // stride + 1
    zoom_scale = img_size / num_patches
    
    # Load model
    network = model.model()
    network.load(".")
    
    # Get the exact image from the path
    image_path = "P0083.jpg"
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
        return
    
    # Process image
    input_image = cv2.imread(image_path)
    input_image = cv2.resize(input_image, (800, 800))  # Ensure consistent size
    img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    classification, attention = network.predict(input_image)

    print(f"Input image shape: {img.shape}")
    print(f"Raw attention shape: {attention.shape}, attention size: {attention.size}")
    
    # Class names for myopic maculopathy grades
    class_names = {
        0: "No myopic maculopathy (Grade 0)",
        1: "Mild myopic maculopathy (Grade 1)",
        2: "Moderate myopic maculopathy (Grade 2)",
        3: "Severe myopic maculopathy (Grade 3)",
        4: "Very severe myopic maculopathy (Grade 4)"
    }
    
    print(f"Classification result: Grade {classification} - {class_names[classification]}")
    
    # Reshape attention map to match the actual size
    attention = attention.reshape(-1)  # Flatten the attention map
    attention_size = int(len(attention) ** 0.5)  # Calculate square root
    if attention_size * attention_size != len(attention):
        # If not a perfect square, find the closest dimensions
        attention_size = int(np.ceil(np.sqrt(len(attention))))
        # Pad the attention map to make it square
        padded_size = attention_size * attention_size
        attention = np.pad(attention, (0, padded_size - len(attention)))
    
    attention = attention.reshape(attention_size, attention_size)
    attention = zoom(attention, zoom=(img_size/attention_size, img_size/attention_size))

    # Plot and save
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.imshow(attention, cmap='jet', alpha=0.3)
    plt.axis('off')
    plt.savefig("attention_visualization.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"Visualization saved as 'attention_visualization.png' in the root folder")


if __name__ == '__main__':
    main()
