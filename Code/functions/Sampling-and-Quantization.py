import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_lab_02(file_path):
    # 1. Load and Convert
    img_bgr = cv2.imread(file_path)
    if img_bgr is None:
        print(f"Error: {file_path} not found.")
        return
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- Part A: Sampling Analysis ---
    scales = [0.25, 0.5, 1.0, 1.5, 2.0]
    fig1, axes1 = plt.subplots(1, 5, figsize=(20, 6))
    
    for i, s in enumerate(scales):
        # Resize
        resized = cv2.resize(img_gray, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        # Scale back to original size for visual comparison of "pixelation"
        h, w = img_gray.shape
        display_img = cv2.resize(resized, (w, h), interpolation=cv2.INTER_NEAREST)
        
        axes1[i].imshow(display_img, cmap='gray')
        axes1[i].set_title(f"Scale: {s}x\nRes: {resized.shape[1]}x{resized.shape[0]}")
        axes1[i].axis('off')
    
    plt.suptitle("6.2.1 Sampling Analysis (Spatial Resolution)", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.show()

    # --- Part B: Quantization Analysis (Bit Depth) ---
    def quantize(image, bits):
        levels = 2 ** bits
        interval = 256 / levels
        return (np.floor(image / interval) * interval).astype(np.uint8)

    bit_depths = [8, 4, 2]
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 6))
    
    for i, b in enumerate(bit_depths):
        quantized_img = quantize(img_gray, b)
        axes2[i].imshow(quantized_img, cmap='gray')
        axes2[i].set_title(f"Bit Depth: {b}-bit\n({2**b} levels)")
        axes2[i].axis('off')
        
    plt.suptitle("6.2.2 Quantization Analysis (Intensity Resolution)", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout(pad=3.0)
    plt.show()

# Run Lab 2
run_lab_02('image1.jpg')