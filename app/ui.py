
import torch
from pathlib import Path
from src.cnn import CNN, img_transform
import torch
from math import exp

model_path = Path(__file__).parent.parent.joinpath("mnist_cnn.pt")
model = CNN()
model.load_state_dict(torch.load(model_path))
model.eval()


def recognize_digit(sketchpad):
    img = sketchpad['layers'][0][:, :, -1]
    img = img_transform(img)
    preds = model(img)
    return {str(i): exp(float(pred)) for i, pred in enumerate(preds[0])}


sp = gr.Sketchpad()

ui = gr.Interface(
    fn=recognize_digit,
    inputs=gr.Sketchpad(),
    outputs=gr.Label(num_top_classes=10),
)
ui.launch()  