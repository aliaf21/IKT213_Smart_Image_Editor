import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser, simpledialog
from pathlib import Path
import io
import os
import ctypes

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk, ImageGrab

from features.filters import gaussian_blur, sobel_edge_detection, binary_threshold, otsu_threshold
from features.history import ImageHistory
from gui.theme import BACKGROUND

PANEL = '#202124'
SURFACE = '#292B30'
HOVER = '#383B42'
TEXT = '#E8E8EA'
MUTED = '#9A9DA5'
ACCENT = '#7695E8'


class MainWindow(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color=BACKGROUND)
        self.pack(fill='both', expand=True)
        self.master = master
        self.image = None
        self.original_image = None
        self.display_image = None
        self.current_path = None
        self.history = ImageHistory()
        self.mode = 'none'
        self.brush_color = '#E8E8EA'
        self.fill_color = '#7695E8'
        self.outline_color = '#E8E8EA'
        self.brush_size = tk.IntVar(value=8)
        self.brush_texture = tk.StringVar(value='Solid')
        self.shape_type = tk.StringVar(value='Rectangle')
        self.zoom = 1.0
        self.view_box = None
        self.selection = None
        self.selection_points = []
        self.selection_kind = None
        self.clipboard = None
        self.drag_start = None
        self.last_point = None
        self.drawing = False
        self.stroke_count = 0
        self.resize_job = None
        self.grid_columnconfigure(0, minsize=215)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.create_toolbar()
        self.create_sidebar()
        self.create_canvas()
        self.create_menu()
        self.create_shortcuts()
        self.update_buttons()

    def create_toolbar(self):
        bar = ctk.CTkFrame(self, height=48, fg_color=PANEL, corner_radius=0)
        bar.grid(row=0, column=0, columnspan=2, sticky='ew')
        ctk.CTkLabel(bar, text='Smart Image Editor', font=('Segoe UI', 16, 'bold'), text_color=TEXT).pack(side='left', padx=20, pady=10)
        self.reset_button = self.toolbar_button(bar, 'Reset', self.reset_image)
        self.redo_button = self.toolbar_button(bar, 'Redo ↪', self.redo)
        self.undo_button = self.toolbar_button(bar, '↶ Undo', self.undo)
        self.mode_label = ctk.CTkLabel(bar, text='View', text_color=MUTED)
        self.mode_label.pack(side='right', padx=12)

    def toolbar_button(self, parent, label, command):
        button = ctk.CTkButton(parent, text=label, command=command, width=75, height=30, fg_color=SURFACE, hover_color=HOVER, text_color=TEXT, corner_radius=6)
        button.pack(side='right', padx=4, pady=8)
        return button

    def section(self, title):
        ctk.CTkLabel(self.sidebar, text=title, font=('Segoe UI', 10, 'bold'), text_color=MUTED, anchor='w').pack(fill='x', padx=14, pady=(18, 6))

    def side_button(self, label, command):
        button = ctk.CTkButton(self.sidebar, text=label, command=command, height=32, fg_color='transparent', hover_color=HOVER, text_color=TEXT, anchor='w', corner_radius=5, font=('Segoe UI', 12))
        button.pack(fill='x', padx=6, pady=2)
        return button

    def create_sidebar(self):
        self.sidebar = ctk.CTkScrollableFrame(self, width=200, fg_color=PANEL, corner_radius=0, scrollbar_button_color=HOVER)
        self.sidebar.grid(row=1, column=0, sticky='nsew', pady=1)
        self.section('FILES')
        self.side_button('New image', self.new_image)
        self.side_button('Open image', self.open_image)
        self.save_button = self.side_button('Save image', self.save_image)
        self.side_button('Save as', self.save_as)
        self.section('IMAGE')
        self.side_button('Rotate right 90°', self.rotate_image)
        self.side_button('Rotate left 90°', self.rotate_left)
        self.side_button('Flip horizontal', self.flip_image)
        self.side_button('Flip vertical', self.flip_vertical)
        self.side_button('Resize', self.resize_image)
        self.side_button('Crop selection', self.crop_selection)
        self.side_button('Grayscale', self.grayscale)
        self.section('SELECTION')
        self.side_button('Rectangle select', lambda: self.set_mode('rectangle'))
        self.side_button('Free-form select', lambda: self.set_mode('lasso'))
        self.side_button('Polygon select', lambda: self.set_mode('polygon'))
        self.side_button('Clear selection', self.clear_selection)
        self.section('TOOLS')
        self.brush_button = self.side_button('Paint brush: OFF', self.toggle_brush)
        self.side_button('Eraser', lambda: self.set_mode('eraser'))
        self.side_button('Color picker', lambda: self.set_mode('picker'))
        self.side_button('Text box', lambda: self.set_mode('text'))
        self.side_button('Zoom in', self.zoom_in)
        self.side_button('Zoom out', self.zoom_out)
        self.side_button('Fit image', self.fit_image)
        self.section('COLORS')
        self.color_button = self.side_button('Brush color', self.choose_brush_color)
        self.color_preview = ctk.CTkLabel(self.sidebar, text='Current brush color', fg_color=self.brush_color, text_color='#000000', corner_radius=5, height=26)
        self.color_preview.pack(fill='x', padx=12, pady=4)
        self.side_button('Shape outline color', self.choose_outline_color)
        self.side_button('Shape fill color', self.choose_fill_color)
        for color in ('#000000', '#FFFFFF', '#E53935', '#43A047', '#1E88E5', '#FDD835'):
            ctk.CTkButton(self.sidebar, text='', width=23, height=23, fg_color=color, hover_color=color, command=lambda c=color: self.set_brush_color(c)).pack(side='top', padx=10, pady=1, anchor='w')
        ctk.CTkLabel(self.sidebar, text='Brush size', text_color=TEXT, anchor='w').pack(fill='x', padx=12, pady=(8, 0))
        ctk.CTkSlider(self.sidebar, from_=1, to=50, number_of_steps=49, variable=self.brush_size, progress_color=ACCENT, button_color=ACCENT).pack(fill='x', padx=12, pady=8)
        ctk.CTkLabel(self.sidebar, text='Brush pattern', text_color=TEXT, anchor='w').pack(fill='x', padx=12)
        ctk.CTkOptionMenu(self.sidebar, variable=self.brush_texture, values=['Solid', 'Dotted', 'Dashed', 'Spray'], fg_color=SURFACE, button_color=HOVER).pack(fill='x', padx=10, pady=6)
        self.section('SHAPES')
        ctk.CTkOptionMenu(self.sidebar, variable=self.shape_type, values=['Rectangle', 'Ellipse', 'Line', 'Triangle'], fg_color=SURFACE, button_color=HOVER).pack(fill='x', padx=10, pady=6)
        self.side_button('Draw shape', lambda: self.set_mode('shape'))
        self.section('FILTERS')
        self.side_button('Gaussian blur', self.apply_blur)
        self.side_button('Sobel edge detection', self.apply_sobel)
        self.side_button('Binary threshold', self.apply_binary)
        self.side_button('Otsu threshold', self.apply_otsu)
        self.section('EXTRA EFFECTS')
        self.side_button('Brightness', self.adjust_brightness)
        self.side_button('Contrast', self.adjust_contrast)
        self.side_button('Sharpen', self.sharpen_image)
        self.side_button('Sepia', self.sepia_image)
        self.side_button('Image histogram', self.show_histogram)

    def create_canvas(self):
        self.image_area = ctk.CTkFrame(self, fg_color=BACKGROUND, corner_radius=0)
        self.image_area.grid(row=1, column=1, sticky='nsew')
        self.image_area.grid_columnconfigure(0, weight=1)
        self.image_area.grid_rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(self.image_area, bg=BACKGROUND, highlightthickness=0, bd=0)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.canvas.bind('<Configure>', self.schedule_preview)
        self.canvas.bind('<ButtonPress-1>', self.on_press)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)
        self.canvas.bind('<Double-Button-1>', self.finish_polygon)
        self.canvas.bind('<ButtonPress-3>', self.finish_polygon)
        self.status = ctk.CTkLabel(self.image_area, text='No image opened', text_color=MUTED, font=('Segoe UI', 11), anchor='w', height=25)
        self.status.grid(row=1, column=0, sticky='ew', padx=15)

    def create_menu(self):
        menubar = tk.Menu(self.master)
        menus = {
            'File': [('New', self.new_image), ('Open', self.open_image), ('Save', self.save_image), ('Save as', self.save_as), ('Properties', self.properties), ('Quit', self.master.destroy)],
            'Clipboard': [('Copy', self.copy_selection), ('Paste', self.paste_selection), ('Cut', self.cut_selection)],
            'Image': [('Rectangular selection', lambda: self.set_mode('rectangle')), ('Free-form selection (Lasso)', lambda: self.set_mode('lasso')), ('Polygon selection', lambda: self.set_mode('polygon')), ('Crop', self.crop_selection), ('Resize', self.resize_image), ('Rotate right 90°', self.rotate_image), ('Rotate left 90°', self.rotate_left), ('Flip vertical', self.flip_vertical), ('Flip horizontal', self.flip_image)],
            'Tools': [('Zoom in', self.zoom_in), ('Zoom out', self.zoom_out), ('Erase', lambda: self.set_mode('eraser')), ('Color picker', lambda: self.set_mode('picker')), ('Paint brush ON/OFF', self.toggle_brush), ('Text box', lambda: self.set_mode('text')), ('Gaussian filter', self.apply_blur), ('Sobel filter', self.apply_sobel), ('Binary filter', self.apply_binary), ('Histogram thresholding (Otsu)', self.apply_otsu)],
            'Shapes': [('Draw shape', lambda: self.set_mode('shape')), ('Select shape', self.choose_shape), ('Outline color', self.choose_outline_color), ('Fill color', self.choose_fill_color)],
            'Colors': [('Color palette', self.choose_brush_color), ('Size of brush', self.set_brush_size)],
            'Effects': [('Brightness', self.adjust_brightness), ('Contrast', self.adjust_contrast), ('Sharpen', self.sharpen_image), ('Sepia', self.sepia_image), ('Histogram', self.show_histogram)],
            'Edit': [('Undo', self.undo), ('Redo', self.redo), ('Reset', self.reset_image)]
        }
        for title, actions in menus.items():
            menu = tk.Menu(menubar, tearoff=0)
            for label, command in actions:
                menu.add_command(label=label, command=command)
            menubar.add_cascade(label=title, menu=menu)
        self.master.configure(menu=menubar)

    def create_shortcuts(self):
        shortcuts = {'<Control-n>': self.new_image, '<Control-o>': self.open_image, '<Control-s>': self.save_image, '<Control-Shift-S>': self.save_as, '<Control-z>': self.undo, '<Control-y>': self.redo, '<Control-c>': self.copy_selection, '<Control-x>': self.cut_selection, '<Control-v>': self.paste_selection, '<Escape>': self.stop_tool, '<b>': self.toggle_brush, '<e>': lambda: self.set_mode('eraser')}
        for sequence, command in shortcuts.items():
            self.master.bind(sequence, lambda event, fn=command: fn())

    def toggle_brush(self):
        self.set_mode('brush')

    def stop_tool(self):
        self.mode = 'none'
        self.selection_points = []
        self.drag_start = None
        self.last_point = None
        self.drawing = False
        self.mode_label.configure(text='View')
        self.brush_button.configure(text='Paint brush: OFF')
        self.canvas.configure(cursor='')
        self.show_image()

    def set_mode(self, mode):
        self.mode = 'none' if self.mode == mode else mode
        self.selection_points = []
        self.last_point = None
        self.mode_label.configure(text=self.mode.capitalize())
        self.brush_button.configure(text='Paint brush: ON' if self.mode == 'brush' else 'Paint brush: OFF')
        self.canvas.configure(cursor='crosshair' if self.mode != 'none' else '')
        self.show_image()

    def set_brush_color(self, color):
        self.brush_color = color
        self.color_preview.configure(fg_color=color, text_color='#FFFFFF' if sum(self.bgr(color)) < 380 else '#000000')

    def choose_brush_color(self):
        result = colorchooser.askcolor(color=self.brush_color, parent=self)
        if result[1]:
            self.set_brush_color(result[1])

    def choose_outline_color(self):
        result = colorchooser.askcolor(color=self.outline_color, parent=self)
        if result[1]:
            self.outline_color = result[1]

    def choose_fill_color(self):
        result = colorchooser.askcolor(color=self.fill_color, parent=self)
        if result[1]:
            self.fill_color = result[1]

    def set_brush_size(self):
        value = simpledialog.askinteger('Brush size', 'Size (1–50):', initialvalue=self.brush_size.get(), minvalue=1, maxvalue=50, parent=self)
        if value is not None:
            self.brush_size.set(value)

    def choose_shape(self):
        choice = simpledialog.askstring('Shape', 'Rectangle, Ellipse, Line or Triangle:', initialvalue=self.shape_type.get(), parent=self)
        if choice and choice.capitalize() in ('Rectangle', 'Ellipse', 'Line', 'Triangle'):
            self.shape_type.set(choice.capitalize())
            self.set_mode('shape')

    def bgr(self, hex_color):
        color = hex_color.lstrip('#')
        return tuple(int(color[i:i + 2], 16) for i in (4, 2, 0))

    def canvas_to_image(self, x, y):
        if self.image is None or self.view_box is None:
            return None
        left, top, shown_width, shown_height = self.view_box
        if not (left <= x < left + shown_width and top <= y < top + shown_height):
            return None
        height, width = self.image.shape[:2]
        return min(width - 1, int((x - left) * width / shown_width)), min(height - 1, int((y - top) * height / shown_height))

    def image_to_canvas(self, point):
        left, top, width, height = self.view_box
        image_height, image_width = self.image.shape[:2]
        return left + point[0] * width / image_width, top + point[1] * height / image_height

    def on_press(self, event):
        point = self.canvas_to_image(event.x, event.y)
        if point is None:
            return
        if self.mode == 'picker':
            blue, green, red = map(int, self.image[point[1], point[0]])
            self.set_brush_color(f'#{red:02x}{green:02x}{blue:02x}')
            return
        if self.mode == 'text':
            self.add_text(point)
            return
        if self.mode == 'polygon':
            self.selection_points.append(point)
            self.draw_overlay()
            return
        self.drag_start = point
        self.last_point = point
        if self.mode == 'lasso':
            self.selection_points = [point]
        if self.mode in ('brush', 'eraser'):
            self.history.save_state(self.image)
            self.drawing = True
            self.stroke_count = 0
            self.draw_stroke(point, point)
            self.show_image()

    def on_drag(self, event):
        if self.drag_start is None or self.image is None:
            return
        point = self.canvas_to_image(event.x, event.y)
        if point is None:
            return
        if self.mode in ('brush', 'eraser') and self.drawing:
            self.draw_stroke(self.last_point, point)
            self.last_point = point
            self.show_image()
        elif self.mode == 'lasso':
            self.selection_points.append(point)
            self.draw_overlay(current=point)
        elif self.mode in ('rectangle', 'shape'):
            self.draw_overlay(current=point)

    def on_release(self, event):
        if self.drag_start is None:
            return
        point = self.canvas_to_image(event.x, event.y)
        if self.mode in ('brush', 'eraser'):
            self.drawing = False
            self.update_buttons()
        elif point is not None and self.mode == 'rectangle':
            x1, y1 = self.drag_start
            x2, y2 = point
            if x1 != x2 and y1 != y2:
                self.selection = self.rectangle_mask(x1, y1, x2, y2)
                self.selection_kind = 'rectangle'
            self.show_image()
        elif point is not None and self.mode == 'lasso':
            self.selection_points.append(point)
            self.create_polygon_selection()
        elif point is not None and self.mode == 'shape':
            self.draw_shape(self.drag_start, point)
        self.drag_start = None
        self.last_point = None

    def rectangle_mask(self, x1, y1, x2, y2):
        height, width = self.image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        mask[top:bottom + 1, left:right + 1] = 255
        return mask

    def create_polygon_selection(self):
        if len(self.selection_points) >= 3:
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(self.selection_points, dtype=np.int32)], 255)
            self.selection = mask
            self.selection_kind = self.mode
        self.selection_points = []
        self.show_image()

    def finish_polygon(self, event=None):
        if self.mode == 'polygon' and self.image is not None:
            self.create_polygon_selection()

    def draw_overlay(self, current=None):
        self.show_image()
        if self.view_box is None:
            return
        if self.mode in ('rectangle', 'shape') and self.drag_start and current:
            x1, y1 = self.image_to_canvas(self.drag_start)
            x2, y2 = self.image_to_canvas(current)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=ACCENT, dash=(5, 3), width=2, tags='overlay')
        elif self.mode in ('lasso', 'polygon') and self.selection_points:
            points = self.selection_points + ([current] if current and self.mode == 'lasso' else [])
            if len(points) >= 2:
                flat = [coordinate for point in points for coordinate in self.image_to_canvas(point)]
                self.canvas.create_line(*flat, fill=ACCENT, width=2, dash=(4, 3), tags='overlay')

    def draw_stroke(self, first, second):
        color = (255, 255, 255) if self.mode == 'eraser' else self.bgr(self.brush_color)
        size = max(1, int(self.brush_size.get()))
        pattern = self.brush_texture.get()
        if self.mode == 'eraser' or pattern == 'Solid':
            cv2.line(self.image, first, second, color, size, cv2.LINE_AA)
            cv2.circle(self.image, second, max(1, size // 2), color, -1, cv2.LINE_AA)
        else:
            distance = max(1, int(np.hypot(second[0] - first[0], second[1] - first[1])))
            for index in range(distance + 1):
                x = int(first[0] + (second[0] - first[0]) * index / distance)
                y = int(first[1] + (second[1] - first[1]) * index / distance)
                self.stroke_count += 1
                if pattern == 'Dotted' and self.stroke_count % max(2, size * 2) != 0:
                    continue
                if pattern == 'Dashed' and (self.stroke_count // max(2, size * 2)) % 2:
                    continue
                if pattern == 'Spray':
                    for _ in range(3):
                        dx, dy = np.random.randint(-size, size + 1, size=2)
                        px, py = x + int(dx), y + int(dy)
                        if 0 <= px < self.image.shape[1] and 0 <= py < self.image.shape[0]:
                            cv2.circle(self.image, (px, py), 1, color, -1)
                else:
                    cv2.circle(self.image, (x, y), max(1, size // 3), color, -1, cv2.LINE_AA)

    def draw_shape(self, start, end):
        if start == end:
            return
        image = self.image.copy()
        x1, x2 = sorted((start[0], end[0]))
        y1, y2 = sorted((start[1], end[1]))
        outline = self.bgr(self.outline_color)
        fill = self.bgr(self.fill_color)
        thickness = max(1, int(self.brush_size.get()))
        kind = self.shape_type.get()
        if kind == 'Rectangle':
            cv2.rectangle(image, (x1, y1), (x2, y2), fill, -1)
            cv2.rectangle(image, (x1, y1), (x2, y2), outline, thickness)
        elif kind == 'Ellipse':
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
            cv2.ellipse(image, center, axes, 0, 0, 360, fill, -1)
            cv2.ellipse(image, center, axes, 0, 0, 360, outline, thickness)
        elif kind == 'Triangle':
            vertices = np.array([[(x1 + x2) // 2, y1], [x1, y2], [x2, y2]], dtype=np.int32)
            cv2.fillPoly(image, [vertices], fill)
            cv2.polylines(image, [vertices], True, outline, thickness)
        else:
            cv2.line(image, start, end, outline, thickness, cv2.LINE_AA)
        self.history.save_state(self.image)
        self.image = image
        self.show_image()

    def add_text(self, point):
        text = simpledialog.askstring('Text box', 'Enter text:', parent=self)
        if not text:
            return
        size = simpledialog.askinteger('Text size', 'Font size:', initialvalue=24, minvalue=8, maxvalue=200, parent=self)
        if size is None:
            return
        scale = size / 30
        image = self.image.copy()
        cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, self.bgr(self.brush_color), max(1, size // 15), cv2.LINE_AA)
        self.history.save_state(self.image)
        self.image = image
        self.show_image()

    def clear_selection(self):
        self.selection = None
        self.selection_points = []
        self.selection_kind = None
        self.show_image()

    def selection_data(self):
        if self.image is None:
            return None
        if self.selection is None:
            return self.image.copy(), np.full(self.image.shape[:2], 255, dtype=np.uint8)
        x, y, width, height = cv2.boundingRect(self.selection)
        return self.image[y:y + height, x:x + width].copy(), self.selection[y:y + height, x:x + width].copy()

    def copy_selection(self):
        data = self.selection_data()
        if data is None:
            return
        self.clipboard = data
        if os.name != 'nt':
            return
        source, mask = data
        rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        rgb[mask == 0] = 255
        picture = Image.fromarray(rgb)
        output = io.BytesIO()
        picture.save(output, format='BMP')
        dib = output.getvalue()[14:]
        kernel32 = ctypes.windll.kernel32
        user32 = ctypes.windll.user32
        kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
        kernel32.GlobalAlloc.restype = ctypes.c_void_p
        kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
        kernel32.GlobalLock.restype = ctypes.c_void_p
        kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
        kernel32.GlobalFree.argtypes = [ctypes.c_void_p]
        user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
        user32.SetClipboardData.restype = ctypes.c_void_p
        handle = kernel32.GlobalAlloc(0x0002, len(dib))
        if not handle:
            return
        address = kernel32.GlobalLock(handle)
        if not address:
            kernel32.GlobalFree(handle)
            return
        ctypes.memmove(address, dib, len(dib))
        kernel32.GlobalUnlock(handle)
        if not user32.OpenClipboard(None):
            kernel32.GlobalFree(handle)
            return
        try:
            if user32.EmptyClipboard() and user32.SetClipboardData(8, handle):
                handle = None
        finally:
            user32.CloseClipboard()
            if handle:
                kernel32.GlobalFree(handle)

    def cut_selection(self):
        if self.image is None:
            return
        self.copy_selection()
        image = self.image.copy()
        if self.selection is None:
            image[:] = 255
        else:
            image[self.selection != 0] = 255
        self.history.save_state(self.image)
        self.image = image
        self.show_image()

    def paste_selection(self):
        if self.image is None:
            return
        external = None
        if os.name == 'nt':
            try:
                external = ImageGrab.grabclipboard()
            except (OSError, ValueError):
                pass
        if isinstance(external, Image.Image):
            rgb = np.array(external.convert('RGB'))
            source = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            mask = np.full(source.shape[:2], 255, dtype=np.uint8)
        elif self.clipboard is not None:
            source, mask = self.clipboard
        else:
            return
        image = self.image.copy()
        x, y = 0, 0
        if self.selection is not None:
            x, y, _, _ = cv2.boundingRect(self.selection)
        height = min(source.shape[0], image.shape[0] - y)
        width = min(source.shape[1], image.shape[1] - x)
        if height <= 0 or width <= 0:
            return
        region = image[y:y + height, x:x + width]
        region[mask[:height, :width] != 0] = source[:height, :width][mask[:height, :width] != 0]
        self.history.save_state(self.image)
        self.image = image
        self.show_image()

    def crop_selection(self):
        if self.image is None or self.selection is None:
            messagebox.showinfo('Crop', 'Select an area first.', parent=self)
            return
        x, y, width, height = cv2.boundingRect(self.selection)
        cropped = self.image[y:y + height, x:x + width].copy()
        cropped[self.selection[y:y + height, x:x + width] == 0] = 255
        self.apply_operation(lambda image: cropped)
        self.selection = None
        self.zoom = 1.0
        self.show_image()

    def resize_image(self):
        if self.image is None:
            return
        height, width = self.image.shape[:2]
        new_width = simpledialog.askinteger('Resize image', 'Width in pixels:', initialvalue=width, minvalue=1, maxvalue=10000, parent=self)
        if new_width is None:
            return
        new_height = simpledialog.askinteger('Resize image', 'Height in pixels:', initialvalue=height, minvalue=1, maxvalue=10000, parent=self)
        if new_height is None:
            return
        self.apply_operation(lambda image: cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA if new_width < width else cv2.INTER_LINEAR))
        self.clear_selection()

    def zoom_in(self):
        self.zoom = min(8.0, self.zoom * 1.25)
        self.show_image()

    def zoom_out(self):
        self.zoom = max(0.1, self.zoom / 1.25)
        self.show_image()

    def fit_image(self):
        self.zoom = 1.0
        self.show_image()

    def properties(self):
        if self.image is None:
            messagebox.showinfo('Properties', 'No image opened.', parent=self)
            return
        height, width = self.image.shape[:2]
        path = self.current_path or 'Untitled'
        messagebox.showinfo('Image properties', f'Name: {Path(path).name}\nWidth: {width} px\nHeight: {height} px\nChannels: {self.image.shape[2]}\nType: {self.image.dtype}\nPath: {path}', parent=self)

    def new_image(self):
        self.image = np.full((600, 800, 3), 255, dtype=np.uint8)
        self.original_image = self.image.copy()
        self.current_path = None
        self.history.clear()
        self.selection = None
        self.selection_points = []
        self.zoom = 1.0
        self.show_image()

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[('Images', '*.png *.jpg *.jpeg *.bmp')], parent=self)
        if not path:
            return
        try:
            with open(path, 'rb') as file:
                data = np.frombuffer(file.read(), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('Invalid image file')
        except (OSError, ValueError, cv2.error) as error:
            messagebox.showerror('Error', str(error), parent=self)
            return
        self.image = image
        self.original_image = image.copy()
        self.current_path = path
        self.history.clear()
        self.selection = None
        self.selection_points = []
        self.zoom = 1.0
        self.show_image()

    def save_image(self):
        if self.image is None:
            return
        if self.current_path:
            self.write_image(self.current_path)
        else:
            self.save_as()

    def save_as(self):
        if self.image is None:
            return
        path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG', '*.png'), ('JPEG', '*.jpg'), ('BMP', '*.bmp')], parent=self)
        if path:
            self.write_image(path)

    def write_image(self, path):
        try:
            extension = Path(path).suffix.lower()
            success, encoded = cv2.imencode(extension, self.image)
            if not success:
                raise ValueError('Save failed')
            with open(path, 'wb') as file:
                file.write(encoded.tobytes())
            self.current_path = path
            self.show_image()
        except (OSError, ValueError, cv2.error) as error:
            messagebox.showerror('Error', str(error), parent=self)

    def apply_operation(self, operation):
        if self.image is None:
            return
        try:
            result = operation(self.image)
            if result is None:
                return
            if result.ndim == 2:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            self.history.save_state(self.image)
            self.image = result
            self.selection = None
            self.show_image()
        except (cv2.error, ValueError, TypeError) as error:
            messagebox.showerror('Error', str(error), parent=self)

    def grayscale(self):
        self.apply_operation(lambda image: cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR))

    def rotate_image(self):
        self.apply_operation(lambda image: cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))

    def rotate_left(self):
        self.apply_operation(lambda image: cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))

    def flip_image(self):
        self.apply_operation(lambda image: cv2.flip(image, 1))

    def flip_vertical(self):
        self.apply_operation(lambda image: cv2.flip(image, 0))

    def apply_blur(self):
        value = simpledialog.askinteger('Gaussian blur', 'Kernel size (odd number):', initialvalue=9, minvalue=1, maxvalue=99, parent=self)
        if value is not None:
            self.apply_operation(lambda image: gaussian_blur(image, value if value % 2 else value + 1))

    def apply_binary(self):
        value = simpledialog.askinteger('Binary threshold', 'Threshold (0–255):', initialvalue=127, minvalue=0, maxvalue=255, parent=self)
        if value is not None:
            self.apply_operation(lambda image: binary_threshold(image, value))

    def apply_sobel(self):
        self.apply_operation(sobel_edge_detection)

    def apply_otsu(self):
        self.apply_operation(otsu_threshold)

    def adjust_brightness(self):
        value = simpledialog.askinteger('Brightness', 'Value (-100 to 100):', initialvalue=20, minvalue=-100, maxvalue=100, parent=self)
        if value is not None:
            self.apply_operation(lambda image: np.clip(image.astype(np.int16) + value, 0, 255).astype(np.uint8))

    def adjust_contrast(self):
        value = simpledialog.askinteger('Contrast', 'Value (0 to 200, 100 = normal):', initialvalue=120, minvalue=0, maxvalue=200, parent=self)
        if value is not None:
            self.apply_operation(lambda image: np.clip((image.astype(np.float32) - 127.5) * (value / 100) + 127.5, 0, 255).astype(np.uint8))

    def sharpen_image(self):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        self.apply_operation(lambda image: cv2.filter2D(image, -1, kernel))

    def sepia_image(self):
        matrix = np.array([[0.131, 0.534, 0.272], [0.168, 0.686, 0.349], [0.189, 0.769, 0.393]])
        self.apply_operation(lambda image: np.clip(cv2.transform(image, matrix), 0, 255).astype(np.uint8))

    def show_histogram(self):
        if self.image is None:
            return
        window = tk.Toplevel(self)
        window.title('Color histogram')
        window.geometry('600x340')
        canvas = tk.Canvas(window, bg='#202124', highlightthickness=0)
        canvas.pack(fill='both', expand=True)
        for channel, color in enumerate(('#5599ff', '#55dd88', '#ff6666')):
            histogram = cv2.calcHist([self.image], [channel], None, [256], [0, 256]).flatten()
            maximum = max(1, float(histogram.max()))
            points = []
            for i, count in enumerate(histogram):
                points.extend((30 + i * 2.1, 300 - float(count) / maximum * 260))
            canvas.create_line(*points, fill=color, width=1.5)
        canvas.create_text(300, 18, text='Blue / Green / Red', fill='#E8E8EA')

    def undo(self):
        if self.image is None:
            return
        previous = self.history.undo(self.image)
        if previous is not None:
            self.image = previous
            self.selection = None
            self.show_image()

    def redo(self):
        if self.image is None:
            return
        next_image = self.history.redo(self.image)
        if next_image is not None:
            self.image = next_image
            self.selection = None
            self.show_image()

    def reset_image(self):
        if self.image is None:
            return
        self.history.save_state(self.image)
        self.image = self.original_image.copy()
        self.selection = None
        self.show_image()

    def schedule_preview(self, event=None):
        if self.resize_job is not None:
            self.after_cancel(self.resize_job)
        self.resize_job = self.after(100, self.show_image)

    def show_image(self):
        self.resize_job = None
        self.canvas.delete('all')
        self.view_box = None
        if self.image is None:
            self.canvas.create_text(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, text='Open an image to begin', fill=MUTED, font=('Segoe UI', 15))
            self.status.configure(text='No image opened')
            self.update_buttons()
            return
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width <= 1 or height <= 1:
            self.after(100, self.show_image)
            return
        image_width, image_height = pil_image.size
        fit = min((width - 40) / image_width, (height - 40) / image_height, 1.0)
        scale = max(0.01, fit * self.zoom)
        shown_width = max(1, int(image_width * scale))
        shown_height = max(1, int(image_height * scale))
        pil_image = pil_image.resize((shown_width, shown_height), Image.Resampling.LANCZOS)
        self.display_image = ImageTk.PhotoImage(pil_image)
        left = width // 2 - shown_width // 2
        top = height // 2 - shown_height // 2
        self.view_box = (left, top, shown_width, shown_height)
        self.canvas.create_image(width // 2, height // 2, anchor='center', image=self.display_image)
        if self.selection is not None:
            contours, _ = cv2.findContours(self.selection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                points = [self.image_to_canvas(tuple(point[0])) for point in contour]
                if len(points) >= 2:
                    flat = [coord for point in points for coord in point]
                    self.canvas.create_line(*flat, *points[0], fill=ACCENT, dash=(4, 3), width=2)
        name = Path(self.current_path).name if self.current_path else 'Untitled'
        self.status.configure(text=f'{image_width} × {image_height} px  ·  {name}  ·  Zoom {int(self.zoom * 100)}%')
        self.update_buttons()

    def update_buttons(self):
        loaded = self.image is not None
        self.save_button.configure(state='normal' if loaded else 'disabled')
        self.reset_button.configure(state='normal' if loaded else 'disabled')
        self.undo_button.configure(state='normal' if loaded and self.history.undo_stack else 'disabled')
        self.redo_button.configure(state='normal' if loaded and self.history.redo_stack else 'disabled')
