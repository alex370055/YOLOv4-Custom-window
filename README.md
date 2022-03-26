- 👋 Hi, I’m @alex370055
- 👀 I’m interested in AI and YOLO
- 🌱 I’m currently learning AI

自述文件
import os
from google.colab import drive
drive.mount('/content/gdrive')
coco 中的文件/mydrive 中的註釋（https://drive.google.com/drive/folders/1G18BWENqpOMPfMxKhLCxSSq5vfHEVtYe?usp=sharing）：

instance_val2017.json

mydrive中yolov4中的文件（https://drive.google.com/drive/folders/1FjZW9NP1hTWR-oXqIgeF2GmzVJfUTMdj?usp=sharing）：

png_to_jpg.py / generate_txt.py / generate_train.py / generate_test.py / test.txt / boxes.csv / obj.zip / obj_test.zip / train_txt.zip / obj.names / weights－best.pt

使用 GPU
第一步：使用 GPU。設置環境。

!sudo apt update
!sudo apt install libgl1-mesa-glx -y
! nvidia-smi
git克隆
Step2：Git克隆項目：https://github.com/WongKinYiu/ScaledYOLOv4

%cd /content/gdrive/My Drive
!wget https://github.com/WongKinYiu/ScaledYOLOv4/archive/yolov4-csp.zip
!unzip yolov4-csp.zip && rm yolov4-csp.zip
安裝要求
Step3：安裝torch==1.6.0+cu101，torchvision==0.7.0+cu101

!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
import os
import cv2
import time
from tqdm import tqdm
from google_drive_downloader import GoogleDriveDownloader as gdd
Step4：Git clone https://github.com/thomasbrandon/mish-cuda，然後安裝。

!git clone https://github.com/thomasbrandon/mish-cuda
%cd mish-cuda
!python setup.py build install
第五步：更新 YAML

!pip install -U PyYAML
Step6：在data文件夾中創建digits.yaml，裡面存放了訓練集、驗證集和測試集的路徑，類別個數和類別名稱。

%cd /content/gdrive/MyDrive/ScaledYOLOv4-yolov4-csp/data
!touch digits.yaml
%cd /content/gdrive/MyDrive/ScaledYOLOv4-yolov4-csp/data

%%writefile digits.yaml
# train: /content/gdrive/MyDrive/ScaledYOLOv4-yolov4-csp/data/train.txt
# val: /content/gdrive/MyDrive/ScaledYOLOv4-yolov4-csp/data/valid.txt
test: /content/gdrive/MyDrive/ScaledYOLOv4-yolov4-csp/data/test.txt
nc: 10
names: ['1.0','2.0','3.0','4.0','5.0','6.0','7.0','8.0','9.0','10.0']
Step7：修改cfg文件。複製一個原始的cfg文件，把圖片的寬高改成576，濾鏡改成45(filters=(classes + 5)*3，classes改成10。

!cp models/yolov4-csp.cfg models/yolov4-csp_416.cfg
!sed -n -e 8p -e 9p -e 1022p -e 1029p -e 1131p -e 1138p -e 1240p -e 1247p models/yolov4-csp_416.cfg

!sed -i '8s/512/576/' models/yolov4-csp_416.cfg
!sed -i '9s/512/576/' models/yolov4-csp_416.cfg
!sed -i '1022s/255/45/' models/yolov4-csp_416.cfg
!sed -i '1029s/80/10/' models/yolov4-csp_416.cfg
!sed -i '1131s/255/45/' models/yolov4-csp_416.cfg
!sed -i '1138s/80/10/' models/yolov4-csp_416.cfg
!sed -i '1240s/255/45/' models/yolov4-csp_416.cfg
!sed -i '1247s/80/10/' models/yolov4-csp_416.cfg
# 查看修改後的參數
!sed -n -e 8p -e 9p -e 1022p -e 1029p -e 1131p -e 1138p -e 1240p -e 1247p models/yolov4-csp_416.cfg
wget測試數據
Step8：上傳所有測試圖片的zip文件（obj_test.zip）。將其解壓縮到 ScaledYOLOv4-yolov4-csp 文件中的數據文件夾中。將 gerenate_test.py 複製到 ScaledYOLOv4-yolov4-csp 文件。然後，運行它。它將生成 test.txt。

運行推理和基準測試
Step9：將weights(best.pt)複製到ScaledYOLOv4-yolov4-csp文件中，將obj.names複製到ScaledYOLOv4-yolov4-csp文件中的data文件夾中。

第十步：測試。

!python test.py --img 576 --conf 0.001 --batch 8 --device 0 --data data/digits.yaml --names data/obj.names --cfg models/yolov4-csp_416.cfg --weights best.pt --task test --save-json
生成 answer.json 以在 Codalab 上提交
Colab 鏈接：https ://colab.research.google.com/drive/1ZydftPlARDwjBYslWqYspbx88jjIczGL?usp=sharing

圖片
