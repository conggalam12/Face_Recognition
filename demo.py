from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import torch
from torch import optim
from torchvision import datasets, transforms
import numpy as np
import os
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
def take_label(str):
    parts = str.split("/")
    file_name_with_extension = parts[-1]
    name_without_extension = file_name_with_extension.split(".")[0]
    if "_" == name_without_extension[-3]:
        print("Label :",name_without_extension[:-3])
    elif "_" == name_without_extension[-4]:
        print("Label :",name_without_extension[:-4])
    else:
        print("Label : ",name_without_extension[:-2])
def load_img(path):
    trans = transforms.Compose([
        transforms.Resize((512, 512)),
    ])
    image = Image.open(path)
    image = trans(image)
    return image
def load_img_res(path):
    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    image = Image.open(path)
    image = trans(image).unsqueeze(0)
    return image
def predict(path):
    take_label(path)
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    image = load_img(path)
    path_img_save = "test.jpg"
    mtcnn(image,path_img_save)

    resnet = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=31
    ).to(device)

    resnet.load_state_dict(torch.load("weights/resnet_face.pth",map_location=torch.device('cpu')))
    resnet.eval()
    image = load_img_res(path_img_save)
    result = resnet(image)
    id = int(torch.argmax(result))

    folder_img = "/home/congnt/congnt/python/face_recognition/img"
    list_name = os.listdir(folder_img)
    list_name.sort()
    with open("class.txt",'a') as file:
        for i in list_name:
            file.write(i+"\n")
    print("Predict:",list_name[id])

predict("img/Vijay Deverakonda/Vijay Deverakonda_7.jpg")