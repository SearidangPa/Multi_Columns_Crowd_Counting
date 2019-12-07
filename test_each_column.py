import os
import torch
import numpy as np
import cv2
import glob
from scipy.io import loadmat
from model_col1 import MCNN_col1
from model_col2 import MCNN_col2
from model_col3 import MCNN_col3


def evalulate_model(model, img_paths):
    mae = 0.0
    mse = 0.0
    for img_path in img_paths:
        #load image 
        #print(img_path)
        img = cv2.imread(img_path,0)
        img = img.reshape((1, 1, img.shape[0], img.shape[1]))
        im_data = torch.from_numpy(img).type(torch.FloatTensor)

        #load labelled groudtruth data 
        GT_path = img_path.replace('.jpg','.mat').replace('images','ground-truth').replace('IMG_','GT_IMG_')
        mat = loadmat(GT_path)
        gt = mat["image_info"][0,0][0,0][0] #2D locations of where the heads are 
        gt_count = len(gt)
        #print(GT_path)
        #print("ground truth: ", len(gt))

        #model's prediction
        estimated_density_map = model(im_data)
        estimated_density_map = estimated_density_map.data.cpu().numpy()
        estimated_count = np.sum(estimated_density_map)
        print("model's prediction: ", estimated_count)

        mae += abs(gt_count-estimated_count)
        mse += ((gt_count-estimated_count)*(gt_count-estimated_count))      
        
    num_samples = len(img_paths)
    mae = mae/num_samples
    mse = np.sqrt(mse/num_samples)

    print("-------------------Results: ----------------------")
    print("mae: ", mae)
    print("mse: ", mse)
    return mae, mse


#get the path of the folder containing the test images
root = os.getcwd()
part_A_test_images = os.path.join(root,'ShanghaiTech/part_A/test_data/images')

#get all the image files in the folder into a list
img_paths = []
for img_path in glob.glob(os.path.join(part_A_test_images, "*.jpg")):
    img_paths.append(img_path)


#paths of pre-trained model 
pre_trained_col1_path = "./pre_trained_model_col1/MCNN_col1_part_A_226.pth"
pre_trained_col2_path = "./pre_trained_model_col2/MCNN_col2_part_A_170.pth"
pre_trained_col3_path = "./pre_trained_model_col3/MCNN_col3_part_A_266.pth"

#load model column 1 
model_col1 = MCNN_col1()
checkpoint1 = torch.load(pre_trained_col1_path)
model_col1.load_state_dict(checkpoint1['state_dict'])
model_col1.eval()   #set to evaluation mode
print("-------------evaluating model column 1------------")
evalulate_model(model_col1, img_paths)

#load model column 2 
model_col2 = MCNN_col2()
checkpoint2 = torch.load(pre_trained_col2_path)
model_col2.load_state_dict(checkpoint2['state_dict'])
model_col2.eval()   #set to evaluation mode
print("-------------evaluating model column 2------------")
evalulate_model(model_col2, img_paths)

#load model column 3
model_col3 = MCNN_col3()
checkpoint3 = torch.load(pre_trained_col3_path)
model_col3.load_state_dict(checkpoint3['state_dict'])
model_col3.eval()               #set to evaluation mode
print("-------------evaluating model column 3------------")
evalulate_model(model_col3, img_paths)









    


