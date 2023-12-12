import gradio as gr
import torch

from src.utils import DataUtils, ModelUtils

cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_available else "cpu")
model = ModelUtils.load_model_from_file("mnist_cnn.pt").to(device)


def recognize_digit(sketchpad):
    img = sketchpad["layers"][0][:, :, -1]
    img = DataUtils.img_transform(img)
    with torch.no_grad():
        img = img.to(device)
        preds = model(img)
        probs = torch.exp(preds)
    return {str(i): float(prob) for i, prob in enumerate(probs)}


ui = gr.Interface(
    fn=recognize_digit,
    inputs=[gr.Sketchpad(crop_size=(28 * 1, 28 * 1))],
    outputs=[gr.Label(num_top_classes=10)],
    title="MNIST Handwriting Recognition",
    description=f"Running On Device: {torch.cuda.get_device_name(0) if cuda_available else None}",
)

if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=7860)
