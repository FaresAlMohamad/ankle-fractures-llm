import gradio as gr
import os
import torch
from monai.networks.nets import DenseNet121
from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    ScaleIntensity,
    Rotate90,
    Resize,
    CropForeground,
    ToTensor,
)
import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description="run a gradio app for ankle fracture classification")
parser.add_argument('--root_dir', type=str, default='C:/Users/fares/PycharmProjects/gradio', help='root_dir that includes the pretrained model')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DenseNet121(spatial_dims=2, in_channels=1,
                    out_channels=1).to(device)
model.load_state_dict(torch.load(os.path.join(args.root_dir, "saved_best_metric_model.pth"), map_location=torch.device(device)))

def threshold_for_cropforeground(x):
    '''helper function for CropForegroundd. Specifies that only pixels with a value bigger than 0.1 are menat to be kept from the edges.'''
    mask = (x >= 0.3) & (x <= 0.7)
    return mask

crop_size = (640, 640)
set_determinism(seed=0)

transforms = Compose([
    ScaleIntensity(),
    CropForeground(source_key="image", allow_smaller=False, select_fn=threshold_for_cropforeground, margin=0),
    Rotate90(k=3),
    Resize(spatial_size=crop_size, mode="area"),
    ToTensor()
])

def get_classification(numpy_array):
    '''returns the classification if given an image as a numpy array'''
    print("numpy_array.shape: ", numpy_array.shape)
    if len(numpy_array.shape) == 3 and numpy_array.shape[2] == 3:
        numpy_array = np.array(Image.fromarray(numpy_array).convert('L'))
    print("numpy_array.shape: ", numpy_array.shape)

    input_tensor = torch.tensor(numpy_array)
    input_tensor = input_tensor.permute(1,0)
    expanded_tensor = input_tensor.unsqueeze(0)

    # Apply the transforms to the array
    input_tensor = transforms(expanded_tensor)

    # Print the shape of the transformed array
    input_tensor = input_tensor.unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        logit = torch.sigmoid(model(input_tensor))
        pred = torch.round(logit)
        final_prediction = int(pred.item())
        answer_texts = [f"The provided image does not include an ankle fracture. Probability of a fracture:{logit.item() * 100: .1f}%",
                        f"The provided image includes an ankle fracture. Probability of a fracture:{logit.item() * 100: .1f}%"]
        return answer_texts[final_prediction]

iface = gr.Interface(
    fn=get_classification,
    inputs=gr.Image(type="numpy", label="Upload Image", image_mode = "L"),
    outputs="text",
    live=True
)

# Launch the interface
iface.launch()