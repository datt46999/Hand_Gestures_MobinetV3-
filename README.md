## 📖 Introduction 
Using MobileNet_V3 for hand gesture.

This model was trained on only 5 classes(ok, call, first, take a picture, dislike) from the 34 classes in the HaGRID v2 dataset.
## 🛠️ Installation
```shell
conda create --name myenv python=3.10 -y
conda activate myenv
git clone https://github.com/datt46999/Hand_Gestures_MobinetV3-.git
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
## 👨‍🏫 Get Started

### Download dataset: Dataset was use from [hagrid_model](https://github.com/hukenovs/hagrid) 


## Train
```shell
python run.py -c train -p configs/MobileNetV3_large.yaml
```

### Test:
```shell 
python run.py -c test -p configs/MobileNetV3_large.yaml
```
## 👀 Model pretrained :
Download: [Model](https://drive.google.com/file/d/1Y0j5G4boUjZ14M6EZSN3piBQmF1L8ZOJ/view?usp=drive_link)

## 👨‍🏫 Deploy and results:
# F1Score = 0.96
# Demo


https://github.com/user-attachments/assets/ec27b97b-2e0d-4ac5-bc99-12d9ef461ac7

