import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_lab_05(file_path):
    # 1. Load and Convert
    img_bgr = cv2.imread(file_path)
    if img_bgr is None:
        print(f"Error: {file_path} not found.")
        return
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # --- 2. Apply Histogram Equalization ---
    img_equ = cv2.equalizeHist(img_gray)
    
    # --- 3. Visualization ---
    plt.figure(figsize=(15, 10))
    
    # Original Image
    plt.subplot(2, 2, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title("Original Grayscale Image")
    plt.axis('off')
    
    # Original Histogram (Fixed Parameter Warning)
    plt.subplot(2, 2, 2)
    plt.hist(img_gray.ravel(), bins=256, range=[0, 256], color='black', alpha=0.7)
    plt.title("Original Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    
    # Equalized Image
    plt.subplot(2, 2, 3)
    plt.imshow(img_equ, cmap='gray')
    plt.title("After Histogram Equalization")
    plt.axis('off')
    
    # Equalized Histogram (Fixed Parameter Warning)
    plt.subplot(2, 2, 4)
    plt.hist(img_equ.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
    plt.title("Equalized Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    
    plt.suptitle("Lab 05: Histogram Processing & Enhancement", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Run Lab 5
run_lab_05('image5.jpg')