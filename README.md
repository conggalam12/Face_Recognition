# Face_Recognition
Fine tune Facenet-pytorch with custom dataset<br>
You can see the original in here [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
## Clone the repository
```
git clone https://github.com/conggalam12/Face_Recognition.git
```
## Install requirement
```
cd Face_Recognition
pip install -r requirements.txt
```
## Download and setup weights
```
mdkir weights
cd weights
wget https://github.com/conggalam12/Face_Recognition/releases/tag/weights/resnet_face.pth
```
## Train with custom dataset
### Setup your dataset
```
|-img
  |-- Name person 1
      |--- image_person_1_1.jpg
      |--- image_person_1_2.jpg
  |-- Name person 2
      |--- image_person_2_1.jpg
      |--- image_person_2_2.jpg
```
### Setup folder train
You set path_folder in exemaple/finetune.ipynb<br>
```
data_dir = 'folder_data_img'
```
And you run train MTCNN , take the face each images <br>
After train the facenet model


    
