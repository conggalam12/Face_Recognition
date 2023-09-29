from models.mtcnn import MTCNN
from models.inception_resnet_v1 import InceptionResnetV1
from models.utils import training
import torch
from torch import optim
from torchvision import datasets, transforms
import numpy as np
import os
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

def load_img(path):
    trans = transforms.Compose([
        transforms.Resize((512, 512)),
    ])
    image = Image.open(path)
    image = trans(image)
    return image


def draw_image(path,name,boxes):
    image = Image.open(path)
    w_image,h_image = image.size
    # image = image.resize((512, 512))
    draw = ImageDraw.Draw(image)
    scale_w = w_image/512
    scale_h = h_image/521
    x, y, w, h = boxes[0][0],boxes[0][1],boxes[0][2],boxes[0][3]
    x , w = x*scale_w,w*scale_w
    y , h = y*scale_h,h*scale_h
    font = ImageFont.truetype("font/times new roman bold.ttf", size=24)
    draw.rectangle([x,y,w,h], outline="green", width=4)
    draw.text((x, y - 30), name, fill="red",font = font)
    image.show()


def predict(path):
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    image = load_img(path)
    faces,batch_boxes= mtcnn(image,None,True)

    resnet = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=31
    ).to(device)

    resnet.load_state_dict(torch.load("weights/resnet_face.pth",map_location=torch.device('cpu')))
    resnet.eval()
    result = resnet(faces.unsqueeze(0))
    id = int(torch.argmax(result))

    folder_img = "/home/congnt/congnt/python/face_recognition/img"
    list_name = os.listdir(folder_img)
    list_name.sort()
    print("Predict:",list_name[id])
    draw_image(path,list_name[id],batch_boxes)


path = "img/Natalie Portman/Natalie Portman_10.jpg"
predict(path)
