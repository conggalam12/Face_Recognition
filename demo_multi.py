from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import torch
from torchvision import transforms
import os
from PIL import Image,ImageDraw,ImageFont
import argparse
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
trans = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])
def load_img(path):
    trans_mtcnn = transforms.Compose([
        transforms.Resize((512, 512)),
    ])
    image = Image.open(path)
    image_trans= trans_mtcnn(image)
    return image , image_trans

def load_box_img(x,y,w,h,image):
    cropped_box = image.crop((x, y, w,h))
    cropped_tensor = trans(cropped_box)
    cropped_tensor = cropped_tensor.unsqueeze(0)
    return cropped_tensor

def draw_image(name,boxes,model,image):
    w_image,h_image = image.size
    draw = ImageDraw.Draw(image)
    scale_w = w_image/512
    scale_h = h_image/521
    for i in range(len(boxes)):
        x, y, w, h = boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]
        image_crop = load_box_img(x,y,w,h,image)
        result = model(image_crop)
        id = int(torch.argmax(result))
        x , w = x*scale_w,w*scale_w
        y , h = y*scale_h,h*scale_h
        font = ImageFont.truetype("font/times new roman bold.ttf", size=20)
        draw.rectangle([x,y,w,h], outline="green", width=4)
        draw.text((x, y - 20), name[id], fill="red",font = font)
    image.show()

def predict(path_img,path_model_resnet):
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    image ,image_trans= load_img(path_img)
    box,_ = mtcnn.detect(image_trans)
    # faces,batch_boxes= mtcnn(image,None,True)

    resnet = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=31
    ).to(device)

    resnet.load_state_dict(torch.load(path_model_resnet,map_location=torch.device('cpu')))
    resnet.eval()
    with open("class.txt",'r') as file:
        name = file.read()
        list_name = name.split("\n")
    draw_image(list_name,box,resnet,image)
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_img',type=str,default='/home/congnt/Pictures/check4.jpg',help='path_img')
    parser.add_argument('--path_model', type=str, default='weights/resnet_face.pth', help='path model resnet')
    opt = parser.parse_args()
    return opt
if __name__ == "__main__":
    opt = parse_opt()
    predict(opt.path_img , opt.path_model)