import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
from pathlib import Path

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk

from features.filters import (
    gaussian_blur,
    sobel_edge_detection,
    binary_threshold,
    otsu_threshold
)
from features.history import ImageHistory
from gui.theme import BACKGROUND


PANEL = "#202124"
SURFACE = "#292B30"
HOVER = "#383B42"
TEXT = "#E8E8EA"
MUTED = "#9A9DA5"
ACCENT = "#7695E8"


class MainWindow(ctk.CTkFrame):

    def __init__(self, master):
        super().__init__(master, fg_color=BACKGROUND)
        self.pack(fill="both", expand=True)

        self.image = None
        self.original_image = None
        self.display_image = None
        self.current_path = None
        self.history = ImageHistory()
        self.resize_job = None
        self.brush_enabled = False
        self.brush_color = "#E8E8EA"
        self.brush_size = tk.IntVar(value=8)
        self.last_brush_point = None

        self.grid_columnconfigure(0, minsize=205)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.create_toolbar()
        self.create_sidebar()
        self.create_canvas()
        self.create_menu(master)
        self.create_shortcuts(master)
        self.update_buttons()

    def create_toolbar(self):
        bar = ctk.CTkFrame(
            self, height=48, fg_color=PANEL,
            corner_radius=0
        )
        bar.grid(
            row=0, column=0, columnspan=2,
            sticky="ew"
        )

        ctk.CTkLabel(
            bar,
            text="Smart Image Editor",
            font=("Segoe UI", 16, "bold"),
            text_color=TEXT
        ).pack(side="left", padx=20, pady=10)

        self.reset_button = self.toolbar_button(
            bar, "Reset", self.reset_image
        )
        self.redo_button = self.toolbar_button(
            bar, "Redo ↪", self.redo
        )
        self.undo_button = self.toolbar_button(
            bar, "↶ Undo", self.undo
        )

    def toolbar_button(self, parent, text, command):
        button = ctk.CTkButton(
            parent,
            text=text,
            command=command,
            width=75,
            height=30,
            fg_color=SURFACE,
            hover_color=HOVER,
            text_color=TEXT,
            corner_radius=6
        )
        button.pack(side="right", padx=4, pady=8)
        return button

    def create_sidebar(self):
        self.sidebar = ctk.CTkScrollableFrame(
            self,
            width=190,
            fg_color=PANEL,
            corner_radius=0,
            scrollbar_button_color=HOVER
        )
        self.sidebar.grid(
            row=1, column=0,
            sticky="nsew", pady=1
        )

        self.sidebar.grid_columnconfigure(0, weight=1)

        self.section("FILES")
        self.new_button = self.side_button(
            "New image", self.new_image
        )
        self.open_button = self.side_button(
            "Open image", self.open_image
        )
        self.save_button = self.side_button(
            "Save image", self.save_image
        )

        self.section("EDIT")
        self.rotate_button = self.side_button(
            "Rotate 90°", self.rotate_image
        )
        self.flip_button = self.side_button(
            "Flip horizontal", self.flip_image
        )
        self.gray_button = self.side_button(
            "Grayscale", self.grayscale
        )

        self.section("PAINT")
        self.brush_button = self.side_button(
            "Paint brush: OFF", self.toggle_brush
        )
        self.color_button = self.side_button(
            "Choose brush color", self.choose_brush_color
        )
        ctk.CTkLabel(
            self.sidebar, text="Brush size", text_color=TEXT,
            anchor="w"
        ).grid(row=self.sidebar.grid_size()[1], column=0,
               sticky="ew", padx=12, pady=(8, 0))
        self.brush_slider = ctk.CTkSlider(
            self.sidebar, from_=1, to=50, number_of_steps=49,
            variable=self.brush_size, progress_color=ACCENT,
            button_color=ACCENT
        )
        self.brush_slider.grid(row=self.sidebar.grid_size()[1],
                               column=0, sticky="ew", padx=12, pady=8)

        self.section("FILTERS")
        self.blur_button = self.side_button(
            "Gaussian blur", self.apply_blur
        )
        self.sobel_button = self.side_button(
            "Edge detection", self.apply_sobel
        )
        self.binary_button = self.side_button(
            "Black & white", self.apply_binary
        )
        self.otsu_button = self.side_button(
            "Auto threshold", self.apply_otsu
        )

        self.settings_frame = ctk.CTkFrame(
            self.sidebar,
            fg_color="transparent"
        )

        self.settings_frame.grid_columnconfigure(
            0, weight=1
        )

        self.settings_title = ctk.CTkLabel(
            self.settings_frame,
            text="",
            font=("Segoe UI", 12, "bold"),
            text_color=TEXT,
            anchor="w"
        )
        self.settings_title.grid(
            row=0, column=0,
            sticky="ew", padx=12, pady=(10, 5)
        )

        self.slider_value = tk.IntVar(value=9)

        self.slider = ctk.CTkSlider(
            self.settings_frame,
            from_=1,
            to=31,
            number_of_steps=15,
            variable=self.slider_value,
            button_color=ACCENT,
            progress_color=ACCENT
        )
        self.slider.grid(
            row=1, column=0,
            sticky="ew", padx=12, pady=10
        )

        self.apply_button = self.side_button(
            "Apply", self.apply_selected_filter,
            parent=self.settings_frame
        )

        self.selected_filter = None

    def section(self, name):
        row = self.sidebar.grid_size()[1]

        ctk.CTkLabel(
            self.sidebar,
            text=name,
            font=("Segoe UI", 10, "bold"),
            text_color=MUTED,
            anchor="w"
        ).grid(
            row=row, column=0,
            sticky="ew",
            padx=14, pady=(22, 8)
        )

    def side_button(self, text, command, parent=None):
        parent = parent or self.sidebar
        row = parent.grid_size()[1]

        button = ctk.CTkButton(
            parent,
            text=text,
            command=command,
            height=32,
            fg_color="transparent",
            hover_color=HOVER,
            text_color=TEXT,
            anchor="w",
            corner_radius=5,
            font=("Segoe UI", 12)
        )
        button.grid(
            row=row, column=0,
            sticky="ew",
            padx=6, pady=2
        )
        return button

    def create_canvas(self):
        self.image_area = ctk.CTkFrame(
            self, fg_color=BACKGROUND,
            corner_radius=0
        )
        self.image_area.grid(
            row=1, column=1, sticky="nsew"
        )

        self.image_area.grid_columnconfigure(
            0, weight=1
        )
        self.image_area.grid_rowconfigure(
            0, weight=1
        )

        self.canvas = tk.Canvas(
            self.image_area,
            bg=BACKGROUND,
            highlightthickness=0,
            bd=0
        )
        self.canvas.grid(
            row=0, column=0, sticky="nsew"
        )

        self.canvas.bind(
            "<Configure>", self.schedule_preview
        )

        self.canvas.bind("<ButtonPress-1>", self.start_brush)
        self.canvas.bind("<B1-Motion>", self.move_brush)
        self.canvas.bind("<ButtonRelease-1>", self.stop_brush)

        self.status = ctk.CTkLabel(
            self.image_area,
            text="No image opened",
            text_color=MUTED,
            font=("Segoe UI", 11),
            anchor="w",
            height=25
        )
        self.status.grid(
            row=1, column=0,
            sticky="ew", padx=15
        )

    def create_menu(self, master):
        menubar = tk.Menu(master)

        menus = {
            "File": [
                ("New", self.new_image),
                ("Open", self.open_image),
                ("Save", self.save_image)
            ],
            "Edit": [
                ("Undo", self.undo),
                ("Redo", self.redo),
                ("Reset", self.reset_image)
            ],
            "Image": [
                ("Rotate 90°", self.rotate_image),
                ("Flip horizontal", self.flip_image)
            ],
            "Filters": [
                ("Grayscale", self.grayscale),
                ("Gaussian blur", self.apply_blur),
                ("Edge detection", self.apply_sobel),
                ("Black & white", self.apply_binary),
                ("Auto threshold", self.apply_otsu)
            ]
        }

        for title, actions in menus.items():
            menu = tk.Menu(menubar, tearoff=0)

            for label, command in actions:
                menu.add_command(
                    label=label, command=command
                )

            menubar.add_cascade(
                label=title, menu=menu
            )

        master.configure(menu=menubar)

    def create_shortcuts(self, master):
        master.bind(
            "<Control-n>", lambda event: self.new_image()
        )
        master.bind(
            "<Control-o>", lambda event: self.open_image()
        )
        master.bind(
            "<Control-s>", lambda event: self.save_image()
        )
        master.bind(
            "<Control-z>", lambda event: self.undo()
        )
        master.bind(
            "<Control-y>", lambda event: self.redo()
        )

    def toggle_brush(self):
        self.brush_enabled = not self.brush_enabled
        self.brush_button.configure(
            text="Paint brush: ON" if self.brush_enabled else "Paint brush: OFF"
        )
        self.canvas.configure(cursor="crosshair" if self.brush_enabled else "")

    def choose_brush_color(self):
        selected = colorchooser.askcolor(color=self.brush_color, parent=self)
        if selected[1]:
            self.brush_color = selected[1]

    def canvas_to_image(self, x, y):
        if self.image is None:
            return None
        height, width = self.image.shape[:2]
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            return None
        scale = min((canvas_width - 40) / width,
                    (canvas_height - 40) / height, 1.0)
        if scale <= 0:
            return None
        shown_width = max(1, int(width * scale))
        shown_height = max(1, int(height * scale))
        left = canvas_width // 2 - shown_width // 2
        top = canvas_height // 2 - shown_height // 2
        if not (left <= x < left + shown_width and
                top <= y < top + shown_height):
            return None
        image_x = min(width - 1, int((x - left) * width / shown_width))
        image_y = min(height - 1, int((y - top) * height / shown_height))
        return image_x, image_y

    def brush_bgr(self):
        color = self.brush_color.lstrip("#")
        red, green, blue = (int(color[i:i + 2], 16) for i in (0, 2, 4))
        return blue, green, red

    def start_brush(self, event):
        if not self.brush_enabled or self.image is None:
            return
        point = self.canvas_to_image(event.x, event.y)
        if point is None:
            return
        self.history.save_state(self.image)
        self.last_brush_point = point
        cv2.circle(self.image, point, max(1, self.brush_size.get() // 2),
                   self.brush_bgr(), -1, lineType=cv2.LINE_AA)
        self.show_image()

    def move_brush(self, event):
        if self.last_brush_point is None or self.image is None:
            return
        point = self.canvas_to_image(event.x, event.y)
        if point is None:
            self.last_brush_point = None
            return
        cv2.line(self.image, self.last_brush_point, point,
                 self.brush_bgr(), max(1, self.brush_size.get()),
                 lineType=cv2.LINE_AA)
        self.last_brush_point = point
        self.show_image()

    def stop_brush(self, event=None):
        self.last_brush_point = None

    def new_image(self):
        self.image = np.full((600, 800, 3), 255, dtype=np.uint8)
        self.original_image = self.image.copy()
        self.current_path = None
        self.history.clear()
        self.last_brush_point = None
        self.show_image()

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp")
            ]
        )

        if not path:
            return

        try:
            with open(path, "rb") as file:
                data = np.frombuffer(
                    file.read(), dtype=np.uint8
                )

            image = cv2.imdecode(
                data, cv2.IMREAD_COLOR
            )

            if image is None:
                raise ValueError("Invalid image file")

        except (OSError, ValueError, cv2.error) as error:
            messagebox.showerror(
                "Error", str(error)
            )
            return

        self.image = image
        self.original_image = image.copy()
        self.current_path = path
        self.history.clear()
        self.last_brush_point = None
        self.show_image()

    def save_image(self):
        if self.image is None:
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg"),
                ("BMP", "*.bmp")
            ]
        )

        if not path:
            return

        try:
            extension = Path(path).suffix.lower()
            success, encoded = cv2.imencode(
                extension, self.image
            )

            if not success:
                raise ValueError("Save failed")

            with open(path, "wb") as file:
                file.write(encoded.tobytes())

        except (OSError, ValueError, cv2.error) as error:
            messagebox.showerror(
                "Error", str(error)
            )

    def apply_operation(self, operation):
        if self.image is None:
            return

        try:
            result = operation(self.image)

            if result is None:
                return

            if result.ndim == 2:
                result = cv2.cvtColor(
                    result, cv2.COLOR_GRAY2BGR
                )

            self.history.save_state(self.image)
            self.image = result
            self.show_image()

        except (cv2.error, ValueError, TypeError) as error:
            messagebox.showerror(
                "Error", str(error)
            )

    def grayscale(self):
        self.apply_operation(
            lambda image: cv2.cvtColor(
                cv2.cvtColor(
                    image, cv2.COLOR_BGR2GRAY
                ),
                cv2.COLOR_GRAY2BGR
            )
        )

    def rotate_image(self):
        self.apply_operation(
            lambda image: cv2.rotate(
                image, cv2.ROTATE_90_CLOCKWISE
            )
        )

    def flip_image(self):
        self.apply_operation(
            lambda image: cv2.flip(image, 1)
        )

    def show_settings(self, filter_name):
        self.selected_filter = filter_name
        self.settings_frame.grid(
            row=self.sidebar.grid_size()[1],
            column=0, sticky="ew",
            padx=6, pady=8
        )

        if filter_name == "blur":
            self.settings_title.configure(
                text="Blur strength"
            )
            self.slider.configure(
                from_=1, to=31,
                number_of_steps=15
            )
            self.slider_value.set(9)

        elif filter_name == "binary":
            self.settings_title.configure(
                text="Threshold"
            )
            self.slider.configure(
                from_=0, to=255,
                number_of_steps=255
            )
            self.slider_value.set(127)

    def apply_selected_filter(self):
        if self.selected_filter == "blur":
            self.apply_operation(
                lambda image: gaussian_blur(
                    image, int(self.slider_value.get())
                )
            )

        elif self.selected_filter == "binary":
            self.apply_operation(
                lambda image: binary_threshold(
                    image, int(self.slider_value.get())
                )
            )

    def apply_blur(self):
        self.show_settings("blur")

    def apply_binary(self):
        self.show_settings("binary")

    def apply_sobel(self):
        self.apply_operation(sobel_edge_detection)

    def apply_otsu(self):
        self.apply_operation(otsu_threshold)

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
        if self.image is None:
            return

        self.history.save_state(self.image)
        self.image = self.original_image.copy()
        self.show_image()

    def schedule_preview(self, event=None):
        if self.resize_job is not None:
            self.after_cancel(self.resize_job)

        self.resize_job = self.after(
            100, self.show_image
        )

    def show_image(self):
        self.resize_job = None
        self.canvas.delete("all")

        if self.image is None:
            width = self.canvas.winfo_width()
            height = self.canvas.winfo_height()

            self.canvas.create_text(
                width // 2,
                height // 2,
                text="Open an image to begin",
                fill=MUTED,
                font=("Segoe UI", 15)
            )
            self.update_buttons()
            return

        rgb = cv2.cvtColor(
            self.image, cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(rgb)

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        if width <= 1 or height <= 1:
            self.after(100, self.show_image)
            return

        image_width, image_height = pil_image.size

        scale = min(
            (width - 40) / image_width,
            (height - 40) / image_height,
            1.0
        )

        new_size = (
            max(1, int(image_width * scale)),
            max(1, int(image_height * scale))
        )

        pil_image = pil_image.resize(
            new_size, Image.Resampling.LANCZOS
        )

        self.display_image = ImageTk.PhotoImage(
            pil_image
        )

        self.canvas.create_image(
            width // 2,
            height // 2,
            anchor="center",
            image=self.display_image
        )

        name = (
            Path(self.current_path).name
            if self.current_path else "Untitled"
        )

        self.status.configure(
            text=f"{image_width} × {image_height} px  ·  {name}"
        )

        self.update_buttons()

    def update_buttons(self):
        loaded = self.image is not None

        buttons = [
            self.save_button,
            self.rotate_button,
            self.flip_button,
            self.gray_button,
            self.blur_button,
            self.sobel_button,
            self.binary_button,
            self.otsu_button,
            self.reset_button,
            self.brush_button,
            self.color_button
        ]

        for button in buttons:
            button.configure(
                state="normal" if loaded else "disabled"
            )

        self.undo_button.configure(
            state="normal"
            if loaded and self.history.undo_stack
            else "disabled"
        )

        self.redo_button.configure(
            state="normal"
            if loaded and self.history.redo_stack
            else "disabled"
        )