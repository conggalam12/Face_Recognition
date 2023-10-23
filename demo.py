from models.mtcnn import MTCNN
from models.inception_resnet_v1 import InceptionResnetV1
from models.utils import training
import torch
from torch import optim
from torchvision import datasets, transforms
import numpy as np
import os
from PIL import Image
import argparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

def load_img(path):
    trans = transforms.Compose([
        transforms.Resize((512, 512)),
    ])
    image = Image.open(path)
    image = trans(image)
    return image


# def draw_image(path,name,boxes):
#     image = Image.open(path)
#     w_image,h_image = image.size
#     # image = image.resize((512, 512))
#     draw = ImageDraw.Draw(image)
#     scale_w = w_image/512
#     scale_h = h_image/521
#     x, y, w, h = boxes[0][0],boxes[0][1],boxes[0][2],boxes[0][3]
#     x , w = x*scale_w,w*scale_w
#     y , h = y*scale_h,h*scale_h
#     font = ImageFont.truetype("font/times new roman bold.ttf", size=24)
#     draw.rectangle([x,y,w,h], outline="green", width=4)
#     draw.text((x, y - 30), name, fill="red",font = font)
#     image.show()


def predict(path_img,path_model_resnet):
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    image = load_img(path_img)
    # faces,batch_boxes= mtcnn(image,None,True)
    faces,_= mtcnn(image,None,True)
    resnet = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=31
    ).to(device)

    resnet.load_state_dict(torch.load(path_model_resnet,map_location=torch.device('cpu')))
    resnet.eval()
    result = resnet(faces.unsqueeze(0))
    id = int(torch.argmax(result))

    with open("class.txt",'r') as file:
        name = file.read()
        list_name = name.split("\n")
    print(list_name[id])
    # draw_image(path_img,list_name[id],batch_boxes)
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_img',type=str,required=True,help='path_img')
    parser.add_argument('--path_model', type=str, default='weights/resnet_face.pth', help='path model resnet')
    opt = parser.parse_args()
    return opt
if __name__ == "__main__":
    opt = parse_opt()
    predict(opt.path_img , opt.path_model)
