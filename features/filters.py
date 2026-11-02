
import cv2
import numpy as np


def validate_image(image):
    """
    Checks that the input is a valid image.
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("A valid image is required.")

    if image.dtype != np.uint8:
        raise ValueError("Image must use uint8 pixel values.")

    if image.ndim == 2:
        return

    if image.ndim == 3 and image.shape[2] in (3, 4):
        return

    raise ValueError("Image must be grayscale, BGR or BGRA.")


def to_grayscale(image):
    """
    Converts an image to grayscale if needed.
    """
    validate_image(image)

    if image.ndim == 2:
        return image.copy()

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_blur(image, kernel_size=15):
    """
    Applies Gaussian blur to reduce noise and soften details.
    """
    validate_image(image)

    if (
        not isinstance(kernel_size, (int, np.integer))
        or isinstance(kernel_size, (bool, np.bool_))
        or kernel_size <= 0
        or kernel_size % 2 == 0
    ):
        raise ValueError("Kernel size must be a positive odd integer.")

    return cv2.GaussianBlur(
        image, (int(kernel_size), int(kernel_size)), 0
    )


def sobel_edge_detection(image, kernel_size=3):
    """
    Detects edges using Sobel filters in X and Y directions.
    """
    gray = to_grayscale(image)

    if (
        not isinstance(kernel_size, (int, np.integer))
        or isinstance(kernel_size, (bool, np.bool_))
        or kernel_size not in (1, 3, 5, 7)
    ):
        raise ValueError("Sobel kernel size must be 1, 3, 5 or 7.")

    sobel_x = cv2.Sobel(
        gray, cv2.CV_64F, 1, 0, ksize=int(kernel_size)
    )

    sobel_y = cv2.Sobel(
        gray, cv2.CV_64F, 0, 1, ksize=int(kernel_size)
    )

    magnitude = cv2.magnitude(sobel_x, sobel_y)

    return cv2.convertScaleAbs(magnitude)


def binary_threshold(image, threshold=127):
    """
    Converts an image to black and white using a fixed threshold.
    """
    gray = to_grayscale(image)

    if (
        not isinstance(threshold, (int, np.integer))
        or isinstance(threshold, (bool, np.bool_))
        or not 0 <= threshold <= 255
    ):
        raise ValueError("Threshold must be an integer from 0 to 255.")

    _, result = cv2.threshold(
        gray, int(threshold), 255, cv2.THRESH_BINARY
    )

    return result


def otsu_threshold(image):
    """
    Automatically finds a threshold using Otsu's method.
    """
    gray = to_grayscale(image)

    _, result = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return result
