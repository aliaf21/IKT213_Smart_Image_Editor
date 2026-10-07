import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


class ImageEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Image Editor")
        self.root.geometry("900x650")

        self.image = None
        self.display_image = None

        self.create_buttons()

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=20)

    def create_buttons(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=10)

        tk.Button(frame, text="Open", command=self.open_image).pack(side="left", padx=5)
        tk.Button(frame, text="Save", command=self.save_image).pack(side="left", padx=5)
        tk.Button(frame, text="Grayscale", command=self.grayscale).pack(side="left", padx=5)
        tk.Button(frame, text="Rotate", command=self.rotate_image).pack(side="left", padx=5)
        tk.Button(frame, text="Flip", command=self.flip_image).pack(side="left", padx=5)

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp")
            ]
        )

        if file_path:
            self.image = cv2.imread(file_path)
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

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        self.show_image()

    def rotate_image(self):
        if self.image is None:
            return

        self.image = cv2.rotate(
            self.image,
            cv2.ROTATE_90_CLOCKWISE
        )

        self.show_image()

    def flip_image(self):
        if self.image is None:
            return

        self.image = cv2.flip(self.image, 1)

        self.show_image()


root = tk.Tk()
app = ImageEditor(root)
root.mainloop()