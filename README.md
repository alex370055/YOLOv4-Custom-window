- ### 👋 Hi, I’m @alex370055
- ### 👀 I’m interested in AI and YOLO
- ### 🌱 I’m currently learning AI

- # Set up the environment.
- ## step1:
- 1.`https://github.com/AlexeyAB/darknet#yolo-v4-v3-and-v2-for-windows-and-linux` download and use.
- 2.you also can use Git to download code
- use cmd:`git clone https://github.com/AlexeyAB/darknet.git`
- ## step2:
- Get Visual Studio ,`https://visualstudio.microsoft.com/zh-hant/downloads/`,The 2019 version is relatively stable.
- ## step3:
- Get CUDA，`https://developer.nvidia.com/cuda-downloads`
- ## step4:
- Get OpenCV，`https://opencv.org/releases/`
- ## step5:
- Get CUDNN， After decompressing the compressed file, copy the data in it to `NVIDIA GPU Computing Toolkit / CUDA / Version Number`.

- # How to train with GPU
- 1.Train it first on 1 GPU for like **1000** iterations: **darknet.exe detector train cfg/tree.data cfg/yolov4tree.cfg yolov4.conv.137**
- 2.Then stop and by using partially-trained model /backup/yolov4_1000.weights run training with multigpu (up to 4 GPUs): darknet.exe detector train cfg/tree.data cfg/yolov4tree.cfg /backup/yolov4_1000.weights
- #  custom parameters
- 1.find `C:\Users\User\Desktop\yolo\darknet-master\build\darknet\x64` you will find some `tree.txt`or`yolov4tree.cfg`
- 2.you need to custom parameters.
- # implement main code
![1](https://user-images.githubusercontent.com/102431773/160269115-89bc9604-bc1b-4501-bede-1d0b3dca723d.JPG)  
you can use `C:\Users\User\Desktop\yolo\darknet-master\build\darknet\x64\yolov4tree.dlh` to Convert length and width and configuration header. 
