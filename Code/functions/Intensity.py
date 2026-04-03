import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_lab_04(file_path):
    # 1. Load and Convert
    img_bgr = cv2.imread(file_path)
    if img_bgr is None:
        print(f"Error: {file_path} not found.")
        return
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # --- a. Negative Transformation ---
    img_negative = 255 - img
    
    # --- b. Log Transformation (Fixed Math) ---
    # Convert to float64 to prevent 8-bit overflow (255+1 = 0 in uint8)
    img_float = img.astype(np.float64)
    max_pixel_val = np.max(img_float)
    
    # Calculate 'c' safely
    if max_pixel_val > 0:
        c = 255 / np.log(1 + max_pixel_val)
        log_transformed = c * (np.log(1 + img_float))
    else:
        log_transformed = img_float # Handle black image
        
    # Clip values and cast back to uint8
    log_transformed = np.clip(log_transformed, 0, 255).astype(np.uint8)
    
    # --- c. Gamma Correction (Power-Law) ---
    def apply_gamma(image, gamma_val):
        # Using a Look-Up Table (LUT) for efficiency
        invGamma = 1.0 / gamma_val
        table = np.array([((i / 255.0) ** invGamma) * 255 
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    gamma_05 = apply_gamma(img, 0.5) # Darkens
    gamma_15 = apply_gamma(img, 1.5) # Brightens
    
    # --- Visualizing Results ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    axes[0, 0].imshow(img, cmap='gray'); axes[0, 0].set_title("Original Grayscale")
    axes[0, 1].imshow(img_negative, cmap='gray'); axes[0, 1].set_title("Negative Transformation")
    axes[0, 2].imshow(log_transformed, cmap='gray'); axes[0, 2].set_title("Log Transformation")
    
    axes[1, 0].imshow(gamma_05, cmap='gray'); axes[1, 0].set_title("Gamma Correction (γ = 0.5)")
    axes[1, 1].imshow(gamma_15, cmap='gray'); axes[1, 1].set_title("Gamma Correction (γ = 1.5)")
    axes[1, 2].axis('off') 
    
    for ax in axes.ravel(): ax.axis('off')
    
    plt.suptitle("Lab 04: Intensity Transformations", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Run Lab 4
run_lab_04('image4.jpg')