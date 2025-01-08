import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import os
from predict import predict_sign

class ModernGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Detector")
        self.root.geometry("700x700")
        self.root.configure(bg='#f0f0f0')  # Light gray background

        # Configure style
        self.style = ttk.Style()
        self.style.configure('Modern.TFrame', background='#f0f0f0')
        self.style.configure('Modern.TButton', padding=10, font=('Helvetica', 12))
        self.style.configure('Modern.TLabel', background='#f0f0f0', font=('Helvetica', 12))

        # Main container
        self.main_frame = ttk.Frame(self.root, style='Modern.TFrame', padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            self.main_frame, 
            text="Traffic Sign Detection", 
            style='Modern.TLabel',
            font=('Helvetica', 24, 'bold')
        )
        title_label.pack(pady=20)

        # Image frame
        self.image_frame = ttk.Frame(self.main_frame, style='Modern.TFrame')
        self.image_frame.pack(pady=20)

        # Image display
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack()

        # Default image display
        self.display_default_image()

        # Buttons frame
        button_frame = ttk.Frame(self.main_frame, style='Modern.TFrame')
        button_frame.pack(pady=20)

        # Buttons
        self.select_btn = ttk.Button(
            button_frame,
            text="Select Image",
            command=self.load_image,
            style='Modern.TButton'
        )
        self.select_btn.pack(side=tk.LEFT, padx=10)

        self.predict_btn = ttk.Button(
            button_frame,
            text="Predict Sign",
            command=self.predict,
            style='Modern.TButton'
        )
        self.predict_btn.pack(side=tk.LEFT, padx=10)

        # Results frame
        results_frame = ttk.Frame(self.main_frame, style='Modern.TFrame')
        results_frame.pack(pady=20, fill=tk.X)

        # Results display
        self.result_var = tk.StringVar()
        self.result_var.set("Select an image and click Predict to analyze the traffic sign")
        
        self.result_label = ttk.Label(
            results_frame,
            textvariable=self.result_var,
            style='Modern.TLabel',
            wraplength=600,
            justify=tk.CENTER
        )
        self.result_label.pack()

        # Initialize variables
        self.current_image_path = None
        self.model_path = "model/traffic_sign_model.h5"

    def display_default_image(self):
        # Create a default image or placeholder
        default_img = Image.new('RGB', (400, 400), '#dddddd')
        photo = ImageTk.PhotoImage(default_img)
        self.image_label.configure(image=photo)
        self.image_label.image = photo

    def load_image(self):
        file_path = filedialog.askopenfilename(
            initialdir="data/test",
            title="Select Traffic Sign Image",
            filetypes=(
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            )
        )
        
        if file_path:
            self.current_image_path = file_path
            try:
                # Load and display image
                image = Image.open(file_path)
                # Maintain aspect ratio while fitting in 400x400
                image.thumbnail((400, 400))
                photo = ImageTk.PhotoImage(image)
                self.image_label.configure(image=photo)
                self.image_label.image = photo
                self.result_var.set("Image loaded. Click Predict to analyze.")
            except Exception as e:
                self.result_var.set(f"Error loading image: {str(e)}")

    def predict(self):
        if not self.current_image_path:
            self.result_var.set("Please select an image first")
            return

        try:
            predicted_class, confidence = predict_sign(self.model_path, self.current_image_path)
            result_text = f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%"
            self.result_var.set(result_text)
        except Exception as e:
            self.result_var.set(f"Error during prediction: {str(e)}")

def main():
    root = tk.Tk()
    app = ModernGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
