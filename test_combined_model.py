import os
import torch
import numpy as np
import cv2
import glob
from scipy.io import loadmat
from model_col1 import MCNN_col1
from model_col2 import MCNN_col2
from model_col3 import MCNN_col3


def evalulate_combined_model(model1, model2, model3, img_paths):
    mae = 0.0
    mse = 0.0
    for img_path in img_paths:
        #load image 
        img = cv2.imread(img_path,0)
        img = img.reshape((1, 1, img.shape[0], img.shape[1]))
        im_data = torch.from_numpy(img).type(torch.FloatTensor)
        #print(img_path)

        #load labelled groudtruth data 
        GT_path = img_path.replace('.jpg','.mat').replace('images','ground-truth').replace('IMG_','GT_IMG_')
        mat = loadmat(GT_path)
        gt = mat["image_info"][0,0][0,0][0] #2D locations of where the heads are 
        gt_count = len(gt)
        #print(GT_path)
        #print("ground truth: ", len(gt))

        #predictions from each of the three columns
        estimated_density_map_model1 = model1(im_data)
        estimated_density_map_model2 = model2(im_data)
        estimated_density_map_model3 = model3(im_data)

        #convert from torch's tensor to numpy
        estimated_density_map_model1 = estimated_density_map_model1.data.cpu().numpy()
        estimated_density_map_model2 = estimated_density_map_model2.data.cpu().numpy()
        estimated_density_map_model3 = estimated_density_map_model3.data.cpu().numpy()

        #get estimated count from each column
        estimated_count_model1 = np.sum(estimated_density_map_model1)
        estimated_count_model2 = np.sum(estimated_density_map_model2)
        estimated_count_model3 = np.sum(estimated_density_map_model3)
        
        #print("model1 prediction: ", estimated_count_model1)
        #print("model2 prediction: ", estimated_count_model2)
        #print("model3 prediction: ", estimated_count_model3)

        estimated_count_combined_model = (estimated_count_model1+estimated_count_model2+estimated_count_model3)/3
        #print("combined_model prediction: ", estimated_count_combined_model)
        
        mae += abs(gt_count-estimated_count_combined_model)
        mse += ((gt_count-estimated_count_combined_model)*(gt_count-estimated_count_combined_model))      
        
    num_samples = len(img_paths)
    mae = mae/num_samples
    mse = np.sqrt(mse/num_samples)

    print("-------------------Results: ----------------------")
    print("mae: ", mae)
    print("mse: ", mse)
    return mae, mse


#get paths of images of the test set
root = os.getcwd()
part_A_test_images = os.path.join(root,'ShanghaiTech/part_A/test_data/images')

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

#load model column 2 
model_col2 = MCNN_col2()
checkpoint2 = torch.load(pre_trained_col2_path)
model_col2.load_state_dict(checkpoint2['state_dict'])
model_col2.eval()   #set to evaluation mode

#load model column 3
model_col3 = MCNN_col3()
checkpoint3 = torch.load(pre_trained_col3_path)
model_col3.load_state_dict(checkpoint3['state_dict'])
model_col3.eval()               #set to evaluation mode


print("-------------evaluating combined model-----------------")
evalulate_combined_model(model_col1, model_col2, model_col3, img_paths)









    


