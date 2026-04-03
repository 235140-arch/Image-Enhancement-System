import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. SETUP: Create Directory Structure
# ==========================================
def setup_folders():
    folders = ['images/input', 'images/output', 'results']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print("✔ Directory structure verified.")

# ==========================================
# 2. LAB FUNCTIONS (Integrated with Save Logic)
# ==========================================

def run_lab_01(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('images/output/lab1_grayscale.jpg', gray)
    print("✔ Lab 1 Complete: Image Acquisition")
    return gray

def run_lab_02(img_gray):
    # Sampling 0.5x
    low_res = cv2.resize(img_gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite('images/output/lab2_sampled_05.jpg', low_res)
    # Quantization 2-bit
    level = 4 # 2^2
    quantized = (np.floor(img_gray / (256/level)) * (256/level)).astype(np.uint8)
    cv2.imwrite('images/output/lab2_quantized_2bit.jpg', quantized)
    print("✔ Lab 2 Complete: Sampling & Quantization")

def run_lab_03(img_gray):
    rows, cols = img_gray.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
    rotated = cv2.warpAffine(img_gray, M, (cols, rows))
    cv2.imwrite('images/output/lab3_transformed.jpg', rotated)
    print("✔ Lab 3 Complete: Geometric Transformations")

def run_lab_04(img_gray):
    # Gamma Correction (Best for Enhancement)
    gamma = 1.5
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    enhanced = cv2.LUT(img_gray, table)
    cv2.imwrite('images/output/lab4_intensity.jpg', enhanced)
    print("✔ Lab 4 Complete: Intensity Transformations")

def run_lab_05(img_gray):
    equ = cv2.equalizeHist(img_gray)
    cv2.imwrite('images/output/lab5_histogram.jpg', equ)
    print("✔ Lab 5 Complete: Histogram Processing")

def run_lab_06_pipeline(img_path):
    # Final Integrated Pipeline
    img = cv2.imread(img_path)
    if img is None: return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    
    # Final Gamma fine-tuning
    table = np.array([((i / 255.0) ** (1/1.2)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    final = cv2.LUT(equ, table)
    
    cv2.imwrite('images/output/lab6_final_system.jpg', final)
    print("✔ Lab 6 Complete: Final System Pipeline executed.")

# ==========================================
# 3. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    setup_folders()
    
    # Define your images (ensure these exist in your folder)
    # If testing in root, use 'image1.jpg'. If following GitHub, use 'images/input/image1.jpg'
    try:
        print("\n--- Starting DIP Enhancement System ---")
        
        # Lab 1: Acquisition & Understanding
        gray1 = run_lab_01('image1.jpg')
        
        # Lab 2: Sampling (using image2)
        img2 = cv2.imread('image2.jpg', 0) # Load as grayscale directly
        if img2 is not None: run_lab_02(img2)
        
        # Lab 3: Geometric (using image3)
        img3 = cv2.imread('image3.jpg', 0)
        if img3 is not None: run_lab_03(img3)
        
        # Lab 4: Intensity (using image4)
        img4 = cv2.imread('image4.jpg', 0)
        if img4 is not None: run_lab_04(img4)
        
        # Lab 5: Histogram (using image5)
        img5 = cv2.imread('image5.jpg', 0)
        if img5 is not None: run_lab_05(img5)
        
        # Lab 6: Full Pipeline (using image6)
        run_lab_06_pipeline('image6.jpg')
        
        print("\n--- All Labs Completed Successfully ---")
        print("Results are stored in: images/output/")
        
    except Exception as e:
        print(f"An error occurred: {e}")