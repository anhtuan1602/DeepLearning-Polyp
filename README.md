# DeepLearning Polyp Segmentation - Quach Tuan Anh 20225469

You can download the checkpoint file from my drive: https://drive.google.com/drive/folders/1JXcMKYkSMISY4_fJBNoXzENfQK4UBY--?usp=sharing

Steps for testing model's result: 
```
git clone https://github.com/anhtuan1602/DeepLearning-Polyp.git

cd DeepLearning-Polyp

<!-- Put the checkpoint file which you have downloaded in main directory >

conda create -n unetenv python=3.10
conda activate unetenv
pip install -r requirements.txt

<!-- Test prediction: for faster tesing, I have already included 10 images from Test set to test/ folder, you can pick one from them -->

python3 infer.py --path test/{image_name}.jpeg

<!-- then check predicted/ folder -->
```