import model
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import zoom
from tqdm import tqdm


def main():
    root_path = "/data_sdb/THUHITCoop/validationset/MICCAI-MMAC23 Validation Set/1. Classification of Myopic Maculopathy/category/"

    img_size = 800
    patch_size = 200
    stride = 100
    
    num_patches = (img_size - patch_size) // stride + 1
    zoom_scale = img_size / num_patches
    
    network =  model.model()
    network.load(".")
    
    for label in range(5):
        label_path = os.path.join(root_path, str(label))
        paths = [os.path.join(label_path, file_name) for file_name in os.listdir(label_path)]
        save_path = f"./results/{label}"
        os.makedirs(save_path, exist_ok=True)     

        for path in tqdm(paths):
            file_name =  os.path.join(save_path, os.path.basename(path))

            if os.path.exists(file_name):
                continue

            input_image = cv2.imread(path)
            img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            _, attention = network.predict(input_image)
            attention = attention.reshape(num_patches, num_patches)
            attention = zoom(attention, zoom=(zoom_scale, zoom_scale))

            plt.cla()
            plt.imshow(img)
            plt.imshow(attention, cmap='jet', alpha=0.3)
            plt.savefig(file_name)


if __name__ == '__main__':
    main()
