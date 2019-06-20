# Absolute Pose Regression
This is a Tensorflow implementation of the following paper 

Synthetic View Generation for AbsolutePose Regression and Image Synthesis, P. Purkait, C. Zhao and C. Zach, BMVC 2019  
http://bmvc2018.org/contents/papers/0221.pdf

For any query email to pulak.isi@gmail.com 

## Usage 

It requires .sift files to train and evaluate the network. One example is given in dataset_new_train.txt and along with few training images in sift_new.zip archive (https://drive.google.com/file/d/1YNQeaFQKY_ALLvaM5iCr4IwUSJRyXFw2/view?usp=sharing). 

To train the network:

    python train.py 

To evaluate: 

    python evaluate.py  
