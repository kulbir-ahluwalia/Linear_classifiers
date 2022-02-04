# Linear_classifiers

## Installing the correct versions of CUDA, PyTorch 
For Dell G15 + RTX 3060 GPU, a driver version of ```Driver Version: 495.29.05``` & ```CUDA Version: 11.5 ``` works with Pytorch 1.10.2.    
    
For a local installation of PyTorch, use Stable 1.10.2 with Linux, Pip, Python, CUDA11.3 chosen at https://pytorch.org/get-started/locally/ 
Use the following command:
```
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
This pytorch is for CUDA 11.3 but works for CUDA 11.5 as well.


## DO not install CUDA 11.6
DO NOT install Driver version 510.39.01 with CUDA version 11.6 according to the following link: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=deb_local

Reboot your computer after making any changes for them to take effect.

For a local installation of PyTorch, use Stable 1.10.2 with Linux, Pip, Python, CUDA11.3 chosen at https://pytorch.org/get-started/locally/ 
Use the following command:
```
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
Surprisngly, Pytorch 1.10.2 works with CUDA version 11.6 as well. For RTX 3060 on Dell G15, the driver version 510.39.01 works with CUDA Version: 11.6. But this driver verion is very UNSTABLE. Computer hangs, crashes and fails to boot half of the time.


