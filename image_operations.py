
import numpy as np
import matplotlib.pyplot as plt

# Define two example matrices (images) manually
image1 = np.array([
    [10, 20, 30, 40, 50],
    [60, 70, 80, 90, 100],
    [110, 120, 130, 140, 150],
    [160, 170, 180, 190, 200],
    [210, 220, 230, 240, 250]
], dtype=np.uint8)

image2 = np.array([
    [5, 10, 15, 20, 25],
    [30, 35, 40, 45, 50],
    [55, 60, 65, 70, 75],
    [80, 85, 90, 95, 100],
    [105, 110, 115, 120, 125]
], dtype=np.uint8)

# Helper to display matrices as images
def show_image(title, img):
    plt.imshow(img, cmap="gray", interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.show()

# 1. Display Original Images
show_image("Original Image 1", image1)
show_image("Original Image 2", image2)

# 2. Sum (Matrix Addition)
sum_image = image1 + image2
show_image("Sum Image", sum_image)

# 3. Subtraction (Matrix Subtraction)
subtracted_image = image1 - image2
show_image("Subtracted Image", subtracted_image)

# 4. Product (Matrix Multiplication Element-Wise)
product_image = image1 * image2
show_image("Product Image", product_image)

# 5. Division (Element-Wise)
division_image = np.divide(image1, (image2 + 1)).astype(np.float32)  # Avoid division by zero
show_image("Division Image", division_image)

# 6. Scaling
def scale_image(img, scale_factor):
    rows, cols = img.shape
    scaled_rows = int(rows * scale_factor)
    scaled_cols = int(cols * scale_factor)
    # Initialize a new matrix for the scaled image
    scaled_image = np.zeros((scaled_rows, scaled_cols), dtype=np.uint8)
    for i in range(scaled_rows):
        for j in range(scaled_cols):
            scaled_image[i, j] = img[int(i / scale_factor), int(j / scale_factor)]
    return scaled_image

scaled_image = scale_image(image1, scale_factor=1.5)
show_image("Scaled Image", scaled_image)

# 7. Translation
def translate_image(img, tx, ty):
    rows, cols = img.shape
    translated_image = np.zeros_like(img)
    for i in range(rows):
        for j in range(cols):
            if 0 <= i + ty < rows and 0 <= j + tx < cols:
                translated_image[i + ty, j + tx] = img[i, j]
    return translated_image

translated_image = translate_image(image1, tx=2, ty=1)
show_image("Translated Image", translated_image)

# 8. Rotation
def rotate_image(img, angle):
    angle_rad = np.deg2rad(angle)
    rows, cols = img.shape
    rotated_image = np.zeros_like(img)
    center_row, center_col = rows // 2, cols // 2
    for i in range(rows):
        for j in range(cols):
            new_i = int((i - center_row) * np.cos(angle_rad) - (j - center_col) * np.sin(angle_rad) + center_row)
            new_j = int((i - center_row) * np.sin(angle_rad) + (j - center_col) * np.cos(angle_rad) + center_col)
            if 0 <= new_i < rows and 0 <= new_j < cols:
                rotated_image[new_i, new_j] = img[i, j]
    return rotated_image

rotated_image = rotate_image(image1, angle=45)
show_image("Rotated Image", rotated_image)

# 9. Shear
def shear_image(img, shx, shy):
    rows, cols = img.shape
    max_offset_x = int(shx * rows)
    max_offset_y = int(shy * cols)
    sheared_image = np.zeros((rows + abs(max_offset_y), cols + abs(max_offset_x)), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            new_i = i + int(shy * j)
            new_j = j + int(shx * i)
            if 0 <= new_i < sheared_image.shape[0] and 0 <= new_j < sheared_image.shape[1]:
                sheared_image[new_i, new_j] = img[i, j]
    return sheared_image

sheared_image = shear_image(image1, shx=0.5, shy=0.5)
show_image("Sheared Image", sheared_image)
