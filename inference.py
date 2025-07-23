import gradio as gr
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("sign-language-digit-model.keras")

# Class labels (digits 0 to 9)
labels = [str(i) for i in range(10)]

def predict(image: Image.Image):
    # Resize and preprocess
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32)
    if img_array.shape[-1] == 4:  # remove alpha if present
        img_array = img_array[..., :3]
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

    # Predict
    preds = model.predict(img_array)[0]
    predicted_digit = np.argmax(preds)

    # Plot probability distribution
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh(labels, preds, color="skyblue")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Prediction Probabilities")
    plt.tight_layout()

    return str(predicted_digit), fig

# Launch app
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=["text", gr.Plot()],
    title="Sign Language Digit Classifier",
    description="Upload a hand-gesture image to classify the digit (0–9).",
)

if __name__ == "__main__":
    demo.launch()
