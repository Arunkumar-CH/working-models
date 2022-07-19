
## create streamlit app

# import required libraries and modules
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import densenet121

import streamlit as st

# define prediction function
def predict(image):
    # load DL model
    model = densenet121(pretrained=True)

    model.eval()

    # load classes
    with open('imagenet_class_index.json', 'r') as f:
        classes = json.load(f)

    # preprocess image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # get prediction
    with torch.no_grad():
        output = model(input_batch)

    pred = torch.nn.functional.softmax(output[0], dim=0).cpu().numpy()

    # return confidence and label
    confidence = round(max(pred)*100, 2)
    label = classes[str(np.argmax(pred))][1]

    return confidence, label

# define image file uploader
image = st.file_uploader("Upload image here")

# define button for getting prediction
if image is not None and st.button("Get prediction"):
    # load image using PIL
    input_image = Image.open(image)

    # show image
    st.image(input_image, use_column_width=True)

    # get prediction
    confidence, label = predict(input_image)

    # print results
    "Model is", confidence, "% confident that this image is of a", label