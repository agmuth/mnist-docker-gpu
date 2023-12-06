
import torch
from pathlib import Path

import torch
import gradio as gr
from math import exp
from src.utils import ModelUtils, DataUtils


model = ModelUtils.load_model_from_file("mnist_cnn.pt") # move into api + config

def recognize_digit(sketchpad):
    img = sketchpad['layers'][0][:, :, -1]
    img = DataUtils.img_transform(img)
    preds = model(img)
    probs = torch.exp(preds)
    return {str(i): float(prob) for i, prob in enumerate(probs[0])}


ui = gr.Interface(
    fn=recognize_digit,
    inputs=gr.Sketchpad(),
    outputs=gr.Label(num_top_classes=10),
)
ui.launch()  