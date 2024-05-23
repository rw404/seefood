import io

import dvc.api
import gradio as gr
import torch

from .augs import get_augmentations_transformations
from .model import UfaNet


def predict(inp, transforms, model, labels):
    transformed_inp = transforms(image=inp)["image"]
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(
            model(transformed_inp.unsqueeze(0))[0], dim=0
        )
        res = {labels[i]: float(prediction[i]) for i in range(3)}
    return (res, inp)


def run_seefood():
    labels = ["twix", "snickers", "orbit"]
    model_info = dvc.api.read(
        str("runs/best.ckpt"),
        repo="https://github.com/rw404/seefood",
        mode="rb",
    )
    _, transforms = get_augmentations_transformations()
    model = UfaNet.load_from_checkpoint(io.BytesIO(model_info))

    demo = gr.Interface(
        fn=lambda inp: predict(inp, transforms, model, labels),
        inputs=gr.Image(type="numpy"),
        outputs=[gr.Label(num_top_classes=3), "image"],
    )
    demo.launch(share=True)
