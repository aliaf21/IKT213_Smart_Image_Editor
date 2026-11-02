
import numpy as np


class ImageHistory:
    def __init__(self, max_steps=20):
        self.undo_stack = []
        self.redo_stack = []
        self.max_steps = max_steps

    def save_state(self, image):
        """Save image before applying an edit."""
        if not isinstance(image, np.ndarray):
            return

        self.undo_stack.append(image.copy())

        if len(self.undo_stack) > self.max_steps:
            self.undo_stack.pop(0)

        self.redo_stack.clear()

    def undo(self, current_image):
        """Restore the previous image."""
        if not self.undo_stack:
            return None

        self.redo_stack.append(current_image.copy())
        return self.undo_stack.pop()

    def redo(self, current_image):
        """Restore an undone edit."""
        if not self.redo_stack:
            return None

        self.undo_stack.append(current_image.copy())
        return self.redo_stack.pop()

    def clear(self):
        """Clear history when opening a new image."""
        self.undo_stack.clear()
        self.redo_stack.clear()
