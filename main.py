import customtkinter as ctk
from gui.main_window import MainWindow


if __name__ == "__main__":
    root = ctk.CTk()
    root.title("Smart Image Editor")
    root.geometry("1180x760")
    root.minsize(850, 550)

    MainWindow(root)

    root.mainloop()