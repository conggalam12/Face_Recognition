from models.mtcnn import MTCNN
from models.inception_resnet_v1 import InceptionResnetV1
from models.utils import training
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

def load_dataset(data_dir):
    workers = 0 if os.name == 'nt' else 8
    dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))
    dataset.samples = [
        (p, p.replace(data_dir, data_dir + '_cropped'))
            for p, _ in dataset.samples
    ]
            
    loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=32,
        collate_fn=training.collate_pil
    )
    return loader

def load_model(device):
    mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
    )
    return mtcnn

def croped_image(data_dir_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    mtcnn = load_model(device)
    loader = load_dataset(data_dir_path)
    for i, (x, y) in enumerate(loader):
        mtcnn(x, save_path=y)
        print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')
    del mtcnn
