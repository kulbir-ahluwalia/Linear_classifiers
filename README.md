# Linear_classifiers

## Installing the correct versions of CUDA, PyTorch 

Install CUDA version 11.6 using the commands here: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=deb_local

Reboot your computer.

For a local installation of PyTorch, use Stable 1.10.2 with Linux, Pip, Python, CUDA11.3 chosen at https://pytorch.org/get-started/locally/ 
Use the following command:
```
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
Surprisngly, Pytorch 1.10.2 works with CUDA version 11.6 as well. For RTX 3060 on Dell G15, the driver version 510.39.01 works with CUDA Version: 11.6.


