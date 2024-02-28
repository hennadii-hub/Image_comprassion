import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Завантаження збережених моделей
c_coder = load_model('c_coder_model.h5')
c_decoder = load_model('c_decoder_model.h5')
CNN = load_model('CNN_model.h5')

coded_image_path = 'coded_image.pkl'  # Шлях до файлу для збереження стисненого зображення

# Функція для збереження стисненого зображення
def save_coded_image(coded_image):
    with open(coded_image_path, 'wb') as f:
        pickle.dump(coded_image, f)

# Функція для завантаження стисненого зображення
def load_coded_image():
    global coded_image
    try:
        with open(coded_image_path, 'rb') as f:
            coded_image = pickle.load(f)
            return True
    except FileNotFoundError:
        return False

coded_image = None
load_coded_image()

def compress_image():
    global original_image_pil, coded_image
    # Перетворення зображення в потрібний формат
    image = original_image_pil.resize((256, 256))  # Розмір вхідного зображення моделі
    image_array = np.array(image.convert('L')) / 255.0  # Перетворення в чорно-біле та нормалізація

    # Стискання зображення
    coded_image = c_coder.predict(np.expand_dims(image_array, axis=0).reshape(1, 256, 256, 1))

    # Видалення попереднього зображення перед відображенням нового
    canvas.delete(tk.ALL)

    # Збереження стисненого зображення
    save_coded_image(coded_image)

    # Відображення стисненого зображення
    compressed_image = (coded_image.reshape(64, 64, 128) * 255).astype(np.uint8)[0]
    compressed_image = Image.fromarray(compressed_image)
    compressed_image = compressed_image.resize((256, 256), Image.ANTIALIAS)  # Розширення зображення
    compressed_image = ImageTk.PhotoImage(compressed_image)
    canvas.create_image(0, 0, anchor='nw', image=compressed_image)
    canvas.compressed_image = compressed_image

def restore_image():
    global coded_image
    # Відновлення зображення зі стисненого зображення
    if coded_image is None:
        messagebox.showerror("Error", "No compressed image found. Please compress an image first.")
        return

    decoded_image = c_decoder.predict(coded_image)

    # Видалення попереднього зображення перед відображенням нового
    canvas.delete(tk.ALL)

    # Перетворення зображення назад у зображення PIL та показ його на екрані
    decoded_image = (decoded_image.reshape(256, 256) * 255).astype(np.uint8)
    decoded_image = Image.fromarray(decoded_image)
    decoded_image = ImageTk.PhotoImage(decoded_image)
    canvas.create_image(0, 0, anchor='nw', image=decoded_image)
    canvas.decoded_image = decoded_image

def save_compressed_image():
    global coded_image
    # Перетворення стисненого зображення назад у зображення PIL та збереження його
    compressed_image = Image.fromarray((coded_image.reshape(64, 64, 128) * 255).astype(np.uint8)[0])
    save_path = filedialog.asksaveasfilename(defaultextension=".jpg")
    if save_path:
        compressed_image.save(save_path)

def open_file():
    global original_image, original_image_pil
    # Відкриття файлу та завантаження його як зображення
    file_path = filedialog.askopenfilename()
    if file_path:
        original_image_pil = Image.open(file_path)
        original_image_pil = original_image_pil.resize((256, 256))  # Зміна розміру для підгонки під модель
        original_image = ImageTk.PhotoImage(original_image_pil)
        canvas.create_image(0, 0, anchor='nw', image=original_image)
        canvas.original_image = original_image

# Створення вікна
root = tk.Tk()
root.title("Image Compression and Restoration")

# Створення полотна для відображення зображень
canvas = tk.Canvas(root, width=256, height=256)
canvas.pack()

# Створення кнопок
compress_button = tk.Button(root, text="Compress Image", command=compress_image)
compress_button.pack(side=tk.LEFT, padx=10, pady=10)

restore_button = tk.Button(root, text="Restore Image", command=restore_image)
restore_button.pack(side=tk.LEFT, padx=10, pady=10)

save_button = tk.Button(root, text="Save Compressed Image", command=save_compressed_image)
save_button.pack(side=tk.LEFT, padx=10, pady=10)

open_button = tk.Button(root, text="Open Image File", command=open_file)
open_button.pack(side=tk.LEFT, padx=10, pady=10)

root.mainloop()
