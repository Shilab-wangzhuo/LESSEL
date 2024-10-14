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
- EfficientNet-->[https://github.com/lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
- UNet-->[https://github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)

```bash
cd tools/Step2_YOLOX/YOLOX
pip install -v -e .

tar -xf tools/Step4_qc/EfficientNet-PyTorch.zip -C tools/Step4_qc/EfficientNet-PyTorch
cd tools/Step4_qc/EfficientNet-PyTorch
pip install efficientnet_pytorch
```

## Usage

1. Download the pre-trained weights

- Download the yolox-x pre-trained model and put the folder in "tools\Step2_YOLOX\YOLOX\YOLOX_weights"

[https://1drv.ms/u/c/194ecbbe03b12717/EYqb1VjNxBNDjgWfqT__PP8BBf7OKnYgj2iFBgOiz-L56g?e=xbqGAk](https://1drv.ms/u/c/194ecbbe03b12717/EYqb1VjNxBNDjgWfqT__PP8BBf7OKnYgj2iFBgOiz-L56g?e=xbqGAk)


- Download the Unet pre-trained model and put the folder in "tools\Step5_cut\Pytorch-UNet-master\checkpoints"

[https://1drv.ms/u/c/194ecbbe03b12717/ES4G-bKiHttKsNPzAbeMXcEBngpmwenAAegtcBKlTzeJuw?e=O7Gdbx](https://1drv.ms/u/c/194ecbbe03b12717/ES4G-bKiHttKsNPzAbeMXcEBngpmwenAAegtcBKlTzeJuw?e=O7Gdbx)


2. Training classification model

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

# trainning
python train_2classify/train.py
```
- Modify the model path in `\tools\Step6_classify\efficient_classify.py`
```python
def make_parser():
    parser = argparse.ArgumentParser("EfficientNet process!")
    parser.add_argument("--test_dir",type=str,default=None)
    parser.add_argument("--save_dir",type=str,default=None)
    parser.add_argument("--ori_img_dir",type=str,default=None)
    parser.add_argument("--threshold_B",type=float,default=0.97) # Determine the threshold for your large-cell model
    parser.add_argument("--threshold_S",type=float,default=0.9)  # Determine the threshold for your small-cell model
    parser.add_argument("--weightB",type=str,default="/your/large-cell/model/path",help="large-cell model path")       # Change it to your large-cell model path
    parser.add_argument("--weightS",type=str,default="/your/small-cell/model/path",help="small-cell model path")     # Change it to your small-cell model path
    parser.add_argument("--imgszB",type=int,default=224,help="test image size")
    parser.add_argument("--imgszS",type=int,default=90,help="test image size")
    # opt=parser.parse_known_args()[0]
    return parser

```

3. Running the Process

- Step-by-step instructions

```bash
# Step 1: Split the whole slide images (WSIs) into non-overlapping patches of 1024×1024 pixels.
python tools\Step1_slide\svs_slide.py -i C202401364.svs -o /output_dir/Step1_slide 

# Step 2: Utilize the YOLOX network to detect single cells within the patch-level images.
python tools\Step2_YOLOX\YOLOX\tools\demo1.py image -n yolox-x -c tools\Step2_YOLOX\YOLOX\YOLOX_weights\best_ckpt.pth --path /output_dir/Step1_slide/C202401364 --save_dir /output_dir/Step2_YOLOX --conf 0.3 --nms 0.5 --tsize 1024 --save_result --device gpu

# Step 3: Extract single-cell images from the patch-level images based on the detection results of the YOLOX model.
python tools\Step3_sc_slide\sc_slide.py /output_dir/Step2_YOLOX/C202401364 /output_dir/Step1_slide/C202401364 /output_dir/Step3_sc_slide

# Step 4: Remove blurry cells, incomplete cells, cell fragments, multicellular clusters, impurities, and cell nuclei.
python tools\Step4_qc\QC.py --test_dir /output_dir/Step3_sc_slide/C202401364  --save_dir /output_dir/Step4_qc/C202401364 

# Step 5: Segment cells from the background in the single-cell images to effectively reduce background interference.
python tools\Step5_cut\Pytorch-UNet-master\predict.py -i /output_dir/Step4_qc/C202401364 -o  /output_dir/Step5_cut

# Step 6: Classify whether the cell is benign or malignant.
python tools\Step6_classify\efficient_classify.py --test_dir /output_dir/Step5_cut/C202401364/json_cut_out   --save_dir /output_dir/Step6_classify/C202401364 --ori_img_dir /output_dir/Step4_qc/C202401364 

```

- Run via Bash (Windows)
```bash
# before you start, make sure you activate your conda environment.
process.bat -i /your/input/svs_file/path -o /output/folder 
# [-o] is optional
```



## Demo

We present partial results of eight samples in the `demo` (including four positive samples and four negative samples).

The input file format and the output directory structure are shown as follows.
```  
Input file: C202401364.svs

Output directory structure:

    -- C202401364   
        - Step1_slide   
            - C202401364             # Non-overlapping patches of 1024×1024 pixels.
        - Step2_YOLOX
            - C202401364             # YOLOX model results.
        - Step3_sc_slide
            - C202401364             # Single-cell images.
        - Step4_qc
            - C202401364             # High-quality single-cell images.
            - C202401364.txt         # Probability values for each cell given by the QC model.
        - Step5_cut
            - C202401364
                - json               # JSON files for each single-cell image.
                - json_cut_out       # High-quality single-cell images with the background removed.
                - vis                # Masks of single-cell images.
        - Step6_classify   
            - C202401364
                - cancer             # Cell images identified as malignant (background removed).
                - cancer_ori         # Cell images identified as malignant (with backgroung).
            - C202401364.txt         # Malignancy probability values for each cell given by the binary classification model.
```

**Note: Due to the large number of image files, we only retained the results of five patch-level images in Steps 1 through 5. The results of Step 6 cover the entire WSI.**

## Process Illustration

1. Step 1: Image Preprocessing

Split the whole slide images (WSIs) into non-overlapping patches of 1024×1024 pixels.

2. Step 2: Single-cell Identification

Utilize the YOLOX network to detect single cells within the patch-level images.

3. Step 3: Single-cell Extraction

Extract single-cell images from the patch-level images based on the detection results of the YOLOX model.

4. Step 4: Quality Control

Remove blurry cells, incomplete cells, cell fragments, multicellular clusters, impurities, and cell nuclei.

5. Step 5: Single-cell Segmentation

Segment cells from the background in the single-cell images to effectively reduce background interference.

6. Step 6: Binary Classification

Classify whether the cell is benign or malignant.