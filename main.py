import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from features.history import ImageHistory

from features.filters import (
    gaussian_blur,
    sobel_edge_detection,
    binary_threshold,
    otsu_threshold
)

class ImageEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Image Editor")
        self.root.geometry("900x650")

        self.image = None
        self.display_image = None
        self.history = ImageHistory()
        self.original_image = None

        self.root.bind("<Control-z>", lambda event: self.undo())
        self.root.bind("<Control-y>", lambda event: self.redo())

        self.create_buttons()

        self.create_filter_menu()
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=20)

    def create_buttons(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=10)

        tk.Button(frame, text="Undo", command=self.undo).pack(side="left", padx=5)
        tk.Button(frame, text="Redo", command=self.redo).pack(side="left", padx=5)
        tk.Button(frame, text="Reset", command=self.reset_image).pack(side="left", padx=5)
    
    def create_filter_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)

        file_menu.add_command(
            label="Open",
            command=self.open_image
        )

        file_menu.add_command(
            label="Save",
            command=self.save_image
        )

        menubar.add_cascade(label="File", menu=file_menu)
        image_menu = tk.Menu(menubar, tearoff=0)

        image_menu.add_command(
            label="Rotate 90°",
            command=self.rotate_image
        )

        image_menu.add_command(
            label="Flip Horizontal",
            command=self.flip_image
        )

        menubar.add_cascade(label="Image", menu=image_menu)
        filters_menu = tk.Menu(menubar, tearoff=0)

        filters_menu.add_command(
            label="Grayscale",
            command=self.grayscale
        )
        filters_menu.add_command(
            label="Blur (Gaussian)",
            command=lambda: self.apply_filter(gaussian_blur)
        )
        filters_menu.add_command(
            label="Edge Detection (Sobel)",
            command=lambda: self.apply_filter(sobel_edge_detection)
        )
        filters_menu.add_command(
            label="Black and White (Threshold)",
            command=lambda: self.apply_filter(binary_threshold)
        )
        filters_menu.add_command(
            label="Auto Threshold (Otsu)",
            command=lambda: self.apply_filter(otsu_threshold)
        )

        menubar.add_cascade(label="Filters", menu=filters_menu)
        self.root.config(menu=menubar)

    def apply_filter(self, filter_function):
        if self.image is None:
            return

        result = filter_function(self.image)

        if result.ndim == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        self.history.save_state(self.image)
        self.image = result
        
        self.show_image()

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp")
            ]
        )

        if file_path:
            self.image = cv2.imread(file_path)

            if self.image is not None:
                self.original_image = self.image.copy()
                self.history.clear()

            self.show_image()

    def show_image(self):
        if self.image is None:
            return

        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        height, width = image_rgb.shape[:2]

        if width > 750 or height > 500:
            scale = min(750 / width, 500 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)

            image_rgb = cv2.resize(
                image_rgb,
                (new_width, new_height)
            )

        image_pil = Image.fromarray(image_rgb)
        self.display_image = ImageTk.PhotoImage(image_pil)

        self.image_label.config(image=self.display_image)

    def save_image(self):
        if self.image is None:
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg")
            ]
        )

        if file_path:
            cv2.imwrite(file_path, self.image)

    def grayscale(self):
        if self.image is None:
            return

        self.history.save_state(self.image)
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        self.show_image()

    def rotate_image(self):
        if self.image is None:
            return

        self.history.save_state(self.image)
        self.image = cv2.rotate(
            self.image,
            cv2.ROTATE_90_CLOCKWISE
        )

        self.show_image()

    def flip_image(self):
        if self.image is None:
            return
        
        self.history.save_state(self.image)
        self.image = cv2.flip(self.image, 1)

        self.show_image()
    
    def undo(self):
        if self.image is None:
            return

        previous = self.history.undo(self.image)

        if previous is not None:
            self.image = previous
            self.show_image()

    def redo(self):
        if self.image is None:
            return

        next_image = self.history.redo(self.image)

        if next_image is not None:
            self.image = next_image
            self.show_image()

    def reset_image(self):
        if self.original_image is None:
            return

        self.history.save_state(self.image)
        self.image = self.original_image.copy()
        self.show_image()


root = tk.Tk()
app = ImageEditor(root)
root.mainloop()