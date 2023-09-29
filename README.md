# Face_Recognition
Fine tune Facenet-pytorch with custom dataset<br>
You can see the original in here [facenet-pytorch](https://github.com/timesler/facenet-pytorch) <br>
I recognition 31 persons , you can see in [here](https://github.com/conggalam12/Face_Recognition/blob/main/class.txt)
## Link dataset 
[Data](https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset?select=Original+Images&fbclid=IwAR0m77Hw6AU7EdLQI1tTE434Bl80PnpUispP3I_ashuYdDJRbPQdpDHIfsc)
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
You set path_folder in train.py<br>
```
data_dir = 'folder_data_img'
```
And you run train MTCNN , take the face each images like that <br>
![img1](https://github.com/conggalam12/Face_Recognition/blob/main/img/Zac%20Efron_90.jpg)
![img2](https://github.com/conggalam12/Face_Recognition/blob/main/img/Vijay%20Deverakonda_90.jpg)

Continue train facenet

## How to use
### Setup path image 
```
predict([path_img])
```
### Run
```
python demo.py
```
### Result
```
Running on device: cpu
Predict: Vijay Deverakonda
```

## Model predict
![img1](https://github.com/conggalam12/Face_Recognition/blob/main/img/Natalie.PNG)
![img2](https://github.com/conggalam12/Face_Recognition/blob/main/img/Natalie2.PNG)
![img3](https://github.com/conggalam12/Face_Recognition/blob/main/img/robert.PNG)
![img3](https://github.com/conggalam12/Face_Recognition/blob/main/img/robert2.PNG)



    
