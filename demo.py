import os
import torch
import numpy as np
import cv2
import glob
from scipy.io import loadmat
from model_col1 import MCNN_col1
from model_col2 import MCNN_col2
from model_col3 import MCNN_col3
from matplotlib import pyplot as plt
from matplotlib import cm as CM
import h5py
#paths of pre-trained model 
pre_trained_col1_path = "./pre_trained_model_col1/MCNN_col1_part_A_226.pth"
pre_trained_col2_path = "./pre_trained_model_col2/MCNN_col2_part_A_170.pth"
pre_trained_col3_path = "./pre_trained_model_col3/MCNN_col3_part_A_266.pth"

#load model column 1 
model_col1 = MCNN_col1()
checkpoint1 = torch.load(pre_trained_col1_path)
model_col1.load_state_dict(checkpoint1['state_dict'])
model_col1.eval()   #set to evaluation mode

#load model column 2 
model_col2 = MCNN_col2()
checkpoint2 = torch.load(pre_trained_col2_path)
model_col2.load_state_dict(checkpoint2['state_dict'])
model_col2.eval()   #set to evaluation mode


#load model column 3
model_col3 = MCNN_col3()
checkpoint3 = torch.load(pre_trained_col3_path)
model_col3.load_state_dict(checkpoint3['state_dict'])
model_col3.eval() #set to evaluation mode



#get the path of the image to be shown
root = os.getcwd()
img_path = os.path.join(root,'ShanghaiTech/part_A/train_data/images_crop/IMG_13.jpg')
den_gt_path = './ShanghaiTech/part_A/train_data/Density_Map_GT_Crop/IMG_13.h5'

#load image 
print(img_path)
img = cv2.imread(img_path,0)
img = img.reshape((1, 1, img.shape[0], img.shape[1]))   #reshape to feed into the model
im_data = torch.from_numpy(img).type(torch.FloatTensor) #convert to torch

#load labelled groudtruth data (matlab file)
GT_path = img_path.replace('.jpg','.mat').replace('images_crop','ground-truth').replace('IMG_','GT_IMG_')
mat = loadmat(GT_path)
gt = mat["image_info"][0,0][0,0][0] #2D locations of where the heads are 
gt_count = len(gt) 

#model's prediction
estimated_density_map = model_col1(im_data)
estimated_density_map = estimated_density_map.data.cpu().numpy() #con
estimated_count = np.sum(estimated_density_map)


print("ground truth: ", len(gt))
print("model's prediction: ", estimated_count)

#display the image
img1 = cv2.imread(img_path)
plt.figure()
plt.imshow(img1)

#read in the ground truth density map and display it
with h5py.File(den_gt_path, 'r') as hf:
    den_gt = hf['density'][:]
plt.figure()
plt.imshow(den_gt, cmap=CM.jet)

#display the density map computed by the model 
estimated_density_map = estimated_density_map[0][0] #rehape into a 2D array
plt.figure()
plt.imshow(estimated_density_map,cmap=CM.jet)
plt.show()














    


