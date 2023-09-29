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
```
python demo.py --path_img [path_your_image] --path_model [path_youe_model]
```
## Result
|||
|-------|-------|
|![img1](https://github.com/conggalam12/Face_Recognition/blob/main/img/Natalie.PNG) | ![img2](https://github.com/conggalam12/Face_Recognition/blob/main/img/Natalie2.PNG) |
|![img3](https://github.com/conggalam12/Face_Recognition/blob/main/img/robert.PNG) | ![img4](https://github.com/conggalam12/Face_Recognition/blob/main/img/robert2.PNG) |

## References

1. David Sandberg's facenet repo: [https://github.com/davidsandberg/facenet](https://github.com/davidsandberg/facenet)

1. F. Schroff, D. Kalenichenko, J. Philbin. _FaceNet: A Unified Embedding for Face Recognition and Clustering_, arXiv:1503.03832, 2015. [PDF](https://arxiv.org/pdf/1503.03832)

1. Q. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman. _VGGFace2: A dataset for recognising face across pose and age_, International Conference on Automatic Face and Gesture Recognition, 2018. [PDF](http://www.robots.ox.ac.uk/~vgg/publications/2018/Cao18/cao18.pdf)

1. D. Yi, Z. Lei, S. Liao and S. Z. Li. _CASIAWebface: Learning Face Representation from Scratch_, arXiv:1411.7923, 2014. [PDF](https://arxiv.org/pdf/1411.7923)

1. K. Zhang, Z. Zhang, Z. Li and Y. Qiao. _Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks_, IEEE Signal Processing Letters, 2016. [PDF](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf)

    
