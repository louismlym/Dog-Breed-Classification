# code is retrieved from https://towardsdatascience.com/create-an-image-classification-web-app-using-pytorch-and-streamlit-f043ddf00c24

import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

def predict(image_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open('data/classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
        for i in range(len(classes)):
            words = classes[i].split('_')
            res = words[0].capitalize()
            for j in range(1, len(words)):
                res += ' ' + words[j].capitalize()
            classes[i] = res


    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=False)
    model.fc = nn.Linear(2048, len(classes))
    state = torch.load('logs/final_model/checkpoint-12.pkl', map_location=device)
    model.load_state_dict(state['net'], strict=True)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)

    out = model(batch_t)
    prob = F.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]

st.set_option('deprecation.showfileUploaderEncoding', False)

st.set_page_config(page_title='Dog Breed Classification')
st.title("Dog Breed Classification")
st.markdown("See [list of 120 breeds](https://github.com/louismlym/Dog-Breed-Classification/blob/main/data/classes.txt) we can classify")
st.write("Please upload an image of a dog, and I will try to classify its breed!")

file_up = st.file_uploader("Upload an image", type=["jpg", "jpeg"])

if file_up is not None:
    image = Image.open(file_up).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    placeholder = st.empty()
    placeholder.write("Hmmm, let me think for a bit..")
    #st.write("Just a second...")
    labels = predict(file_up)
    placeholder.empty()

    # print out the top 5 prediction labels with scores
    st.write("I think it is..")
    for i in labels:
        st.write("-", i[0], "| confidence:", i[1], "%")#We think it is {} (confidence: {:.2f}%)".format(i[0], i[1]))