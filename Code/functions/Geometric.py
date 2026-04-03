def run_lab_03(file_path):
    # 1. Load and Convert
    img_bgr = cv2.imread(file_path)
    if img_bgr is None:
        print(f"Error: {file_path} not found.")
        return
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    rows, cols = img.shape
    
    # --- 1. Rotation (45°) ---
    M_rot = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
    rotated = cv2.warpAffine(img, M_rot, (cols, rows))
    # Inverse Rotation (-45°)
    M_rot_inv = cv2.getRotationMatrix2D((cols/2, rows/2), -45, 1)
    restored_rot = cv2.warpAffine(rotated, M_rot_inv, (cols, rows))

    # --- 2. Translation ---
    M_trans = np.float32([[1, 0, 100], [0, 1, 50]])
    translated = cv2.warpAffine(img, M_trans, (cols, rows))
    # Inverse Translation
    M_trans_inv = np.float32([[1, 0, -100], [0, 1, -50]])
    restored_trans = cv2.warpAffine(translated, M_trans_inv, (cols, rows))

    # --- 3. Shearing ---
    M_shear = np.float32([[1, 0.2, 0], [0, 1, 0]])
    sheared = cv2.warpAffine(img, M_shear, (int(cols*1.2), rows))

    # --- Visualizing Results ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Row 1: The Transformations
    axes[0, 0].imshow(rotated, cmap='gray'); axes[0, 0].set_title("45° Rotation")
    axes[0, 1].imshow(translated, cmap='gray'); axes[0, 1].set_title("Translation (+100, +50)")
    axes[0, 2].imshow(sheared, cmap='gray'); axes[0, 2].set_title("Shearing (0.2)")
    
    # Row 2: The Restoration Attempts
    axes[1, 0].imshow(restored_rot, cmap='gray'); axes[1, 0].set_title("Restored Rotation (Inverse)")
    axes[1, 1].imshow(restored_trans, cmap='gray'); axes[1, 1].set_title("Restored Translation (Inverse)")
    axes[1, 2].imshow(img, cmap='gray'); axes[1, 2].set_title("Original (Reference)")
    
    for ax in axes.ravel(): ax.axis('off')
    
    plt.suptitle("Lab 03: Geometric Transformations & Inverse Restoration", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Run Lab 3
run_lab_03('image3.jpg')