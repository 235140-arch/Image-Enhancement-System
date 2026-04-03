import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_lab_01(file_path):
    # 1. Load image (RGB/Grayscale)
    # Note: OpenCV loads as BGR, so we convert to RGB for Matplotlib
    img_bgr = cv2.imread(file_path)
    
    if img_bgr is None:
        print(f"Error: Could not find '{file_path}'. Please ensure it is uploaded.")
        return

    # 2. Convert RGB to Grayscale
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 3. Data Inspection (The "Understanding" part)
    print("="*30)
    print("  INITIAL IMAGE REPORT")
    print("="*30)
    print(f"Resolution (H x W x C): {img_bgr.shape}")
    print(f"Data Type:             {img_bgr.dtype}")
    print(f"Min Pixel Value:       {np.min(img_gray)}")
    print(f"Max Pixel Value:       {np.max(img_gray)}")
    print("-"*30)
    print("Partial Matrix (Top-left 5x5 Grayscale):")
    print(img_gray[:5, :5])
    print("="*30)

    # 4. Display
    plt.figure(figsize=(12, 6))
    
    # Original RGB
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image (RGB)")
    plt.axis('off')

    # Grayscale
    plt.subplot(1, 2, 2)
    plt.imshow(img_gray, cmap='gray')
    plt.title("Converted Image (Grayscale)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Run the Lab
run_lab_01('image2.jpg')