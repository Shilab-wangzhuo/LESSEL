#pip install efficientnet_pytorch
#为efficientnet训练分类的数据进行预处理（训练集切分+补边）
import os
import glob
import cv2
import random
from pathlib import Path


from PIL import Image
from shutil import move

def filter_and_move_images(source_folder, destination_B, destination_S, min_size=90):
    # 确保目标文件夹存在
    os.makedirs(destination_B, exist_ok=True)
    os.makedirs(destination_S, exist_ok=True)
    
    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)
        
        try:
            # 打开图片
            with Image.open(file_path) as img:
                width, height = img.size
                
                # 检查宽度或高度是否大于90px
                if width > min_size or height > min_size:
                    # 移动符合条件的图片到目标文件夹
                    dest_path = os.path.join(destination_B, filename)
                    move(file_path, dest_path)
                    
                else:
                    dest_path = os.path.join(destination_S, filename)
                    move(file_path, dest_path)
                    

        except IOError:
            print(f'Moved: {filename} ')
        except Exception as e:
            print(f'Error processing {filename}: {e}')




#补边,这一步主要是为了将图片填充为正方形，防止直接resize导致图片变形
def expend_img(img):
    '''
    :param img: 图片数据
    :return:
    '''
    fill_pix=[255,255,255] #填充色素，可自己设定
    h,w=img.shape[:2]
    if h>=w: #左右填充
        padd_width=int(h-w)//2
        padd_top,padd_bottom,padd_left,padd_right=0,0,padd_width,padd_width #各个方向的填充像素
    elif h<w: #上下填充
        padd_high=int(w-h)//2
        padd_top,padd_bottom,padd_left,padd_right=padd_high,padd_high,0,0 #各个方向的填充像素
    new_img = cv2.copyMakeBorder(img,padd_top,padd_bottom,padd_left,padd_right,cv2.BORDER_CONSTANT, value=fill_pix)
    return new_img


#切分训练集和测试集，并进行补边处理
def split_train_test(img_dir,save_dir,train_val_num):
    '''
    :param img_dir: 原始图片路径，注意是所有类别所在文件夹的上一级目录
    :param save_dir: 保存图片路径
    :param train_val_num: 切分比例
    :return:
    '''
    img_dir_list=glob.glob(img_dir+os.sep+"*")#获取每个类别所在的路径（一个类别对应一个文件夹）
    for class_dir in img_dir_list:
        class_name=class_dir.split(os.sep)[-1] #获取当前类别
        img_list=glob.glob(class_dir+os.sep+"*") #获取每个类别文件夹下的所有图片
        all_num=len(img_list) #获取总个数
        train_list=random.sample(img_list,int(all_num*train_val_num)) #训练集图片所在路径
        save_train=save_dir+os.sep+"train"+os.sep+class_name
        save_val=save_dir+os.sep+"val"+os.sep+class_name
        os.makedirs(save_train,exist_ok=True)
        os.makedirs(save_val,exist_ok=True) #建立对应的文件夹
        print(class_name+" trian num",len(train_list))
        print(class_name+" val num",all_num-len(train_list))
        #保存切分好的数据集
        for imgpath in img_list:
            imgname=Path(imgpath).name #获取文件名
            if imgpath in train_list:
                img=cv2.imread(imgpath)
                new_img=expend_img(img)
                cv2.imwrite(save_train+os.sep+imgname,new_img)
            else: #将除了训练集意外的数据均视为验证集
                img = cv2.imread(imgpath)
                new_img = expend_img(img)
                cv2.imwrite(save_val + os.sep + imgname, new_img)

    print("split train and val finished !")

if __name__ == '__main__':
    
    source_cancer = 'train_2classify/data/cancer'
    source_normal = 'train_2classify/data/normal'
    cancerB = 'train_2classify/dataB/cancerB'
    normalB = 'train_2classify/dataB/normalB'
    cancerS = 'train_2classify/dataS/cancerS'
    normalS = 'train_2classify/dataS/normalS'
    os.makedirs(cancerB,exist_ok=True)
    os.makedirs(normalB,exist_ok=True)
    os.makedirs(normalS,exist_ok=True)
    os.makedirs(cancerS,exist_ok=True)
    filter_and_move_images(source_cancer, cancerB, cancerS)
    filter_and_move_images(source_normal, normalB, normalS)
    dataB = 'train_2classify/dataB'
    dataS = 'train_2classify/dataS'
    datasetB="train_2classify/datasetB"
    datasetS="train_2classify/datasetS"
    train_val_num=0.78
    split_train_test(dataB,datasetB,train_val_num)
    split_train_test(dataS,datasetS,train_val_num)
