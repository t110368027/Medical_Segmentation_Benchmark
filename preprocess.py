import os
import argparse
import cv2
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Grayscale, Normalize, ToTensor
from tqdm import tqdm

def data_process(data_path, name, mode):
    if name == "DRIVE":
        if mode == "test":
            return 
        else:
            img_path = os.path.join(data_path, mode, "images")
            gt_path = os.path.join(data_path, mode, "1st_manual")
            file_list = list(sorted(os.listdir(img_path)))
    elif name == "CHASEDB1":
        img_path = os.path.join(data_path, "Images")
        gt_path = os.path.join(data_path, "Masks")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "STARE":
        img_path = os.path.join(data_path, "imgs")
        gt_path = os.path.join(data_path, "label-ah")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "STARE_u":
        img_path = data_path
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "DCA1":
        data_path = os.path.join(data_path, "Database_134_Angiograms")
        file_list = list(sorted(os.listdir(data_path)))
    elif name == "CHUAC":
        img_path = os.path.join(data_path, "Original")
        gt_path = os.path.join(data_path, "Photoshop")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "HRF":
        img_path = os.path.join(data_path, "images")
        gt_path = os.path.join(data_path, "manual1")
        file_list = list(sorted(os.listdir(img_path)))
    img_list = []
    gt_list = []
    img_list_name = []
    for i, file in tqdm(enumerate(file_list)):
        img_list_name.append(file)
        if name == "DRIVE":
            return
            img = Image.open(os.path.join(img_path, file))
            gt = Image.open(os.path.join(gt_path, file[0:2] + "_manual1.gif"))
            img = Grayscale(1)(img)
            img_list.append(ToTensor()(img).numpy())
            gt_list.append(ToTensor()(gt).numpy())
        elif name == "HRF":
            return
            if mode == "training" and int(file[0:2]) <= 10:
                img = Image.open(os.path.join(img_path, file))
                print(os.path.join(img_path, file))
                gt = Image.open(os.path.join(gt_path, file.split('.')[0] + '.tif'))
                img = Grayscale(1)(img)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))
            elif mode == "test" and int(file[0:2]) > 10:
                img = Image.open(os.path.join(data_path, file))
                gt = Image.open(os.path.join(data_path, file.split('.')[0] + '.tif'))
                img = Grayscale(1)(img)
                img_list.append(ToTensor()(img).numpy())
                gt_list.append(ToTensor()(gt).numpy())
        elif name == "CHASEDB1":
            # if len(file) == 13:
            if mode == "training" and int(file[6:8]) <= 10:
                img = Image.open(os.path.join(img_path, file))
                gt = Image.open(os.path.join(
                    gt_path, file[0:9] + '_1stHO.png'))
                img = Grayscale(1)(img)
                img_list.append(ToTensor()(img).numpy())
                gt_list.append(ToTensor()(gt).numpy())
            elif mode == "test" and int(file[6:8]) > 10:
                img = Image.open(os.path.join(img_path, file))
                gt = Image.open(os.path.join(
                    gt_path, file[0:9] + '_1stHO.png'))
                img = Grayscale(1)(img)
                img_list.append(ToTensor()(img).numpy())
                gt_list.append(ToTensor()(gt).numpy())
        elif name == "DCA1":
            if len(file) <= 7:
                if mode == "training" and int(file[:-4]) <= 100:
                    img = cv2.imread(os.path.join(data_path, file), 0)
                    gt = cv2.imread(os.path.join(
                        data_path, file[:-4] + '_gt.pgm'), 0)
                    gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                    img_list.append(ToTensor()(img).numpy())
                    gt_list.append(ToTensor()(gt).numpy())
                elif mode == "test" and int(file[:-4]) > 100:
                    img = cv2.imread(os.path.join(data_path, file), 0)
                    gt = cv2.imread(os.path.join(
                        data_path, file[:-4] + '_gt.pgm'), 0)
                    gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                    img_list.append(ToTensor()(img).numpy())
                    gt_list.append(ToTensor()(gt).numpy())
        elif name == "CHUAC":
            if mode == "training" and int(file[:-4]) <= 20:
                img = cv2.imread(os.path.join(img_path, file), 0)
                if int(file[:-4]) <= 17 and int(file[:-4]) >= 11:
                    tail = "PNG"
                else:
                    tail = "png"
                gt = cv2.imread(os.path.join(
                    gt_path, "angio"+file[:-4] + "ok."+tail), 0)
                gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                img = cv2.resize(
                    img, (512, 512), interpolation=cv2.INTER_LINEAR)
                img_list.append(ToTensor()(img).numpy())
                gt_list.append(ToTensor()(gt).numpy())
            elif mode == "test" and int(file[:-4]) > 20:
                img = cv2.imread(os.path.join(img_path, file), 0)
                gt = cv2.imread(os.path.join(
                    gt_path, "angio"+file[:-4] + "ok.png"), 0)
                gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                img = cv2.resize(
                    img, (512, 512), interpolation=cv2.INTER_LINEAR)
                img_list.append(ToTensor()(img).numpy())
                gt_list.append(ToTensor()(gt).numpy())
        elif name == "STARE":
            if not file.endswith("gz"):
                img = Image.open(os.path.join(img_path, file))
                gt = Image.open(os.path.join(gt_path, file[0:6] + '.ah.png'))
                img = Grayscale(1)(img)
                gt = Grayscale(1)(gt)
                img_list.append(ToTensor()(img).numpy())
                gt_list.append(ToTensor()(gt).numpy())
        elif name == "STARE_u":
            if not file.endswith("gz"):
                img = Image.open(os.path.join(img_path, file))
                img = Grayscale(1)(img)
                img_list.append(ToTensor()(img).numpy())     
    img_list = normalization(img_list)
    img_list = np.array(img_list).transpose(0,2,3,1)
    img_list_name = np.array(img_list_name)
    
    if name == "STARE_u":
        return img_list, None, img_list_name
    gt_list = np.array(gt_list).transpose(0,2,3,1)
    assert img_list.shape,gt_list.shape
    
    return img_list, gt_list, img_list_name

def normalization(imgs_list):
    imgs_list = torch.from_numpy(np.array(imgs_list))
    imgs = imgs_list
    mean = torch.mean(imgs)
    std = torch.std(imgs)
    normal_list = []
    for i in imgs_list:
        n = Normalize([mean], [std])(i)
        n = (n - torch.min(n)) / (torch.max(n) - torch.min(n))
        normal_list.append(n.numpy())
    return normal_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_path', default="data", type=str,
                        help='the path of dataset',required=True)
    parser.add_argument('-dn', '--dataset_name', default="DRIVE", type=str,
                        help='the name of dataset',choices=['CHASEDB1','STARE','CHUAC','DCA1','STARE_u'],required=True)
    args = parser.parse_args()
    
    if args.dataset_name == "DRIVE":
        args.dataset_path = os.path.join(args.dataset_path,"DRIVE")
    elif args.dataset_name == "CHASEDB1":
        args.dataset_path = os.path.join(args.dataset_path,"CHASE_DB1")
    elif args.dataset_name == "CHUAC":
        args.dataset_path = os.path.join(args.dataset_path,"angiography")
    elif args.dataset_name == "DCA1":
        args.dataset_path = os.path.join(args.dataset_path,"DB_Angiograms_134")
    elif args.dataset_name == "STARE":
        args.dataset_path = os.path.join(args.dataset_path,"STARE")
    elif args.dataset_name == "STARE_u":
        args.dataset_path = os.path.join(args.dataset_path,"STARE-img")
    elif args.dataset_name == "HRF":
        args.dataset_path = os.path.join(args.dataset_path,"HRF")
    
    if not os.path.exists("./datasets/{}".format(args.dataset_name)):
        os.makedirs("./datasets/{}".format(args.dataset_name))
    
    train_img, train_gt, train_name = data_process(args.dataset_path, args.dataset_name, "training")
    val_img, val_gt, val_name = data_process(args.dataset_path, args.dataset_name, "test")
    
    if args.dataset_name == "STARE":
        c = 1
        print(train_name)
        for i in range(0,20,2):
            n_val_img, n_val_gt, n_val_name = [], [], []
            
            n_train_img = np.delete(train_img,(i,i+1), axis=0)
            n_train_gt = np.delete(train_gt,(i,i+1), axis=0)
            n_train_name = np.delete(train_name,(i,i+1), axis=0)
            
            n_val_img.append(val_img[i]),n_val_img.append(val_img[i+1])
            n_val_gt.append(val_gt[i]),n_val_gt.append(val_gt[i+1])
            n_val_name.append(val_name[i]),n_val_name.append(val_name[i+1])
            
            n_val_img, n_val_gt, n_val_name = np.array(n_val_img), np.array(n_val_gt), np.array(n_val_name) 
            np.savez_compressed('./datasets/{}/set{}.npz'.format(args.dataset_name,c),
                                x_train=n_train_img, y_train=n_train_gt,
                                x_val=n_val_img, y_val=n_val_gt,
                                train_name=n_train_name, val_name=n_val_name
                                )
            c+=1
            
    elif args.dataset_name == "STARE_u":  
         np.savez_compressed('./datasets/{}/set.npz'.format(args.dataset_name),
                            image=train_img,image_name=train_name,
                            )
    else:
        np.savez_compressed('./datasets/{}/set.npz'.format(args.dataset_name),
                            x_train=train_img, y_train=train_gt,
                            x_val=val_img, y_val=val_gt,
                            train_name=train_name, val_name=val_name
                            )
    