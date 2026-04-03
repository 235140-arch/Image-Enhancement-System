import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def process_image(input_path):
    """
    Complete DIP Enhancement Pipeline:
    Grayscale -> Histogram Equalization -> Gamma Correction
    """
    # 1. Image Acquisition
    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        print(f"Error: {input_path} not found.")
        return None
    
    # 2. Pre-processing: Convert to Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 3. Step 1: Histogram Equalization (Fixes Contrast)
    equ = cv2.equalizeHist(gray)
    
    # 4. Step 2: Gamma Correction (γ = 1.2 for highlighting details)
    # Using the LUT method for professional performance
    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    enhanced_image = cv2.LUT(equ, table)
    
    return gray, equ, enhanced_image

# --- Running the Final Pipeline ---
input_file = 'image6.jpg'
original_gray, step1_equ, final_enhanced = process_image(input_file)

if final_enhanced is not None:
    # Save the output to your project structure
    os.makedirs('images/output', exist_ok=True)
    cv2.imwrite('images/output/enhanced_result.jpg', final_enhanced)
    
    # Visual Comparison for the Report
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_gray, cmap='gray')
    plt.title("1. Original Grayscale")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(step1_equ, cmap='gray')
    plt.title("2. After Hist. Equalization")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(final_enhanced, cmap='gray')
    plt.title("3. Final Enhanced (Hist + Gamma)")
    plt.axis('off')
    
    plt.suptitle("Lab 06: Final Integrated Enhancement System Pipeline", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("Success! Enhanced image saved to 'images/output/enhanced_result.jpg'")