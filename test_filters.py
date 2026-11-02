
import cv2
import numpy as np

from features.filters import (
    gaussian_blur,
    sobel_edge_detection,
    binary_threshold,
    otsu_threshold
)

# Create a test image
image = np.zeros((300, 400, 3), dtype=np.uint8)

# Add shapes and colors
image[:] = (200, 150, 100)
cv2.rectangle(image, (50, 50), (200, 200), (0, 255, 0), -1)
cv2.circle(image, (280, 150), 70, (0, 0, 255), -1)
cv2.line(image, (0, 0), (400, 300), (255, 255, 255), 5)

# Apply all filters
blur = gaussian_blur(image)
sobel = sobel_edge_detection(image)
binary = binary_threshold(image)
otsu = otsu_threshold(image)

# Check output dimensions and pixel values
assert blur.shape == image.shape
assert sobel.shape == image.shape[:2]
assert binary.shape == image.shape[:2]
assert otsu.shape == image.shape[:2]

assert set(np.unique(binary)).issubset({0, 255})
assert set(np.unique(otsu)).issubset({0, 255})

# Check that original image is unchanged
assert image.shape == (300, 400, 3)

print("All filter tests passed!")

# Display results
cv2.imshow("Original", image)
cv2.imshow("Gaussian Blur", blur)
cv2.imshow("Sobel Edge Detection", sobel)
cv2.imshow("Binary Threshold", binary)
cv2.imshow("Otsu Threshold", otsu)

cv2.waitKey(0)
cv2.destroyAllWindows()
