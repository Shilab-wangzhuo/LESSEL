# LESSEL: Lung cancer detection with single-cell sequencing and deep learning

## Quick start

Before you start, we recommend you to create a new conda environment. 

1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch 1.13 or later](https://pytorch.org/get-started/locally/)

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Install model framework

References:
- YOLOX-->[https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- EfficientNet-->[https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
- UNet-->[https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)

```bash
cd tools/Step2_YOLOX/YOLOX
pip install -v -e .

cd tools/Step4_qc/EfficientNet-PyTorch
pip install efficientnet_pytorch
```

## Usage

1. Download the pre-trained weights

- Download the yolox-x pre-trained model and put the folder in "tools\Step2_YOLOX\YOLOX\YOLOX_weights"

[https://1drv.ms/f/c/194ecbbe03b12717/EsPvowajTnRJtxs7KCSsBhEBL58k-0rNUXW7hICLGnH_-A?e=gM5ExA](https://1drv.ms/f/c/194ecbbe03b12717/EsPvowajTnRJtxs7KCSsBhEBL58k-0rNUXW7hICLGnH_-A?e=gM5ExA)


- Download the Unet pre-trained model and put the folder in "tools\Step5_cut\Pytorch-UNet-master\checkpoints"

[https://1drv.ms/f/c/194ecbbe03b12717/EvP2fynFlkhJlKY0dyD1Gz8Bl4V4X858KBu7sUZtWKhudw?e=of8pnI](https://1drv.ms/f/c/194ecbbe03b12717/EvP2fynFlkhJlKY0dyD1Gz8Bl4V4X858KBu7sUZtWKhudw?e=of8pnI)

2. Run the process

```bash
# before you start, make sure you activate your conda environment.
process.bat -i /your/input/svs_file/path -o /output/folder 
```

3. Training classification model

- Put your training images in 'data' folder, creating two folders to seperate the images in two catagories.

e.g. 
```  
    -- data   
        - positive   
        - negative   
```

- Run the following commands 
```bash
# preparing dataset
python train_2classify/prepare_dataset.py

#trainning
python train_2classify/train.py
```


