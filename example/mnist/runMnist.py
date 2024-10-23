import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import pickle

# dimension of the MNIST image
WIDTH, HEIGHT = 28, 28

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST-Zahl malen")
        self.canvas = tk.Canvas(root, width=280, height=280, bg='black')
        self.canvas.pack()

        # Event listener for drawing
        self.canvas.bind("<B1-Motion>", self.paint)
        
        # Buttons for clearing and predicting
        self.button_clear = tk.Button(root, text="Delete", command=self.clear_canvas)
        self.button_clear.pack()
        self.button_save = tk.Button(root, text="Predict", command=self.save_image)
        self.button_save.pack()

        # PIL image and drawing tool
        self.image = Image.new("L", (280, 280), 'black')
        self.draw = ImageDraw.Draw(self.image)
        
    def paint(self, event):
        # Draw with white color to match the MNIST format
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
        self.draw.ellipse([x1, y1, x2, y2], fill='white')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 'black')
        self.draw = ImageDraw.Draw(self.image)
        
    def save_image(self):
        # Resize image to 28x28
        image_resized = self.image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize to values between 0 (black) and 1 (white)
        image_array = np.array(image_resized) / 255.0
        image_vector = image_array.flatten()  # transform to a vector

        # Show image optionaly
        # plt.imshow(image_array, cmap='gray')
        # plt.show()

        # import model
        with open('example/mnist/mnist.pkl', 'rb') as file:
            mlp = pickle.load(file)

        # Print image vector
        print("Bild als Vektor:")
        print(image_vector)

        # Predict image
        vector = image_vector.reshape(-1, 1)
        output = mlp.forward(vector)
        # print all predictions round to 4 decimal places
        print("Prediction:", np.round(output.get_y_pred(), 3))
        print("Prediction:", np.argmax(output.get_y_pred()))

# Create the drawing app
root = tk.Tk()
app = DrawingApp(root)
root.mainloop()
