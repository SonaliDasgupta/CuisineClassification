from flask import Flask, url_for, request, render_template
import requests
from io import BytesIO
from PIL import Image
from path import Path
import urllib
import urllib.request
import numpy as np
import os
import torch
from torch import nn
from torchvision import datasets, transforms, models


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def predict():
    if request.method == 'POST':
        url = request.form['location']
        response = requests.get(url)
        test_transforms = transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])
        ])

        classes = ['Indian', 'Italian', 'Mexican']
        model = torch.load('cuisinemodel.pth')
        urllib.request.urlretrieve(url, "local1.jpg")
        image = Image.open(Path('local1.jpg'))
        input = test_transforms(image)
        input = input.view(1, 3, 224, 224)
        output = model(input)
        prediction = torch.argmax(output.data)
    return render_template('results.html', prediction = prediction)

if __name__=='__main__':
    app.run(host = '127.0.0.1', port = 5000, debug = True)

   
