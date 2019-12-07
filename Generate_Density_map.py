import os
import csv
import cv2
import math
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as CM
from scipy.io import loadmat
from scipy import spatial
from scipy.ndimage.filters import gaussian_filter 
import scipy
import PIL.Image as Image
import h5py
import glob
import natsort

#read h5py file 
def Read_Density_Map(img_name):
    with h5py.File(img_name, 'r') as hf:
        data = hf['density'][:]
        plt.figure()
        plt.imshow(data,cmap=CM.jet)
        plt.show()


def Generate_Crop_Density():
    root = os.getcwd()      #get current directory 

    #for part A of the training dataset
    part_A_train_path_density_map = os.path.join(root,'ShanghaiTech/part_A/train_data/Density_Map_GT')
    part_A_train_path_images = os.path.join(root,'ShanghaiTech/part_A/train_data/images')

    #path names for the density files 
    density_paths = []
    for density_path in glob.glob(os.path.join(part_A_train_path_density_map, "*.h5")):
        density_paths.append(density_path)

    #create an output directory
    DenMap_crop_dir = os.path.join(root,'ShanghaiTech/part_A/train_data/Density_Map_GT_Crop')
    if not os.path.exists(DenMap_crop_dir):
        os.mkdir(DenMap_crop_dir)

    #create an output directory
    image_crop_dir = os.path.join(root,'ShanghaiTech/part_A/train_data/images_crop')
    if not os.path.exists(image_crop_dir):
        os.mkdir(image_crop_dir)
    
    #for each density file 
    for density_path in density_paths:
        
        #find the corresponding image file for each density file 
        img_path = density_path.replace('Density_Map_GT', 'images').replace('h5', 'jpg')
        im = cv2.imread(img_path, 0)    #read and convert to grayscale 

        with h5py.File(density_path, 'r') as hf:
            data = hf['density'][:]
    
            h, w = data.shape
            den1 = data[0:(h-math.floor(h/2)), 0:(w-math.floor(w/2))]  #top left
            den2 = data[(h-math.floor(h/2)):h, 0:(w-math.floor(w/2))]  #bottom left
            den3 = data[0:(h-math.floor(h/2)), (w-math.floor(w/2)):w]  #top right
            den4 = data[(h-math.floor(h/2)):h, (w-math.floor(w/2)):w]  #bottom right
            den_arr = [den1,den2,den3,den4]

            im1 = im[0:(h-math.floor(h/2)), 0:(w-math.floor(w/2))]  #top left
            im2 = im[(h-math.floor(h/2)):h, 0:(w-math.floor(w/2))]  #bottom left
            im3 = im[0:(h-math.floor(h/2)), (w-math.floor(w/2)):w]  #top right
            im4 = im[(h-math.floor(h/2)):h, (w-math.floor(w/2)):w]  #bottom right
            im_arr = [im1,im2,im3,im4]
            
            #generate 5 random crops of size 1/9 of the original images
            for i in range(0,5):
                y = random.randint(0,math.floor(h*2/3))     #random num in height
                x = random.randint(0,math.floor(w*2/3))     #random num in width      
                im_arr.append(im[y:(y+math.floor(h/3)),x:(x+math.floor(w/3))])
                den_arr.append(data[y:(y+math.floor(h/3)),x:(x+math.floor(w/3))])

            for i in range(len(im_arr)):
                DenMap_crop_save_path = density_path.replace('Density_Map_GT', 'Density_Map_GT_Crop').replace('.h5', '_'+str(i+1)+'.h5')
                im_crop_save_path = img_path.replace('images', 'images_crop').replace('.jpg', '_'+str(i+1)+'.jpg')
                
                print(DenMap_crop_save_path)
                print(im_crop_save_path)
                
                with h5py.File(DenMap_crop_save_path, 'w') as hf:
                    hf['density'] = den_arr[i]
                
                cv2.imwrite(im_crop_save_path, im_arr[i])
                
        hf.close()
        


#gt is a 2D array of all 0's with the exception of places of the labelled heads equal to 1
#TO-Do: check to see if we can do without list(zip(...))
def gaussian_filter_density(gt):
    #creat a place holder for the result which is the density map
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt) #count the number of people in the image

    #if no human in the image, then return density of 0's
    if gt_count == 0:   
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    
    # build kdtree to find the nearest neighbors for each point efficiently
    leafsize = 2048
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    
    # query kdtree to get the distances to the 3 nearest neighbors for each pt
    # the locations variable is just a placeholder. k=4 but this include the point itself
    distances, locations = tree.query(pts, k=4)


    #for each labelled head
    for i, pt in enumerate(pts):
        #create an empty 2D array of the same dimenstion as the image so we can calculate 
        #the contribution of the density each labelled head to the total density 
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1
        
        if gt_count > 1:
            # sigma takes into account the geometric distortion factor 
            # calculated by a constant times the average of the distances to its 3 nearest neighbors
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.08
        
        else: #case: 1 point: sigma =  (average of the two dimensions)/4. 
            # width of the kernel w = 2*int(truncate*sigma + 0.5) + 1. trunicate = 4 by default
            sigma = np.average(np.array(gt.shape))/4. 
        print(i)
        # add the contribution of the density the pt to the total density 
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density

def Generate_Density_Map_Batch():
    root = os.getcwd()
    part_A_train = os.path.join(root,'ShanghaiTech/part_A/train_data/images')
    
    #create an output directory to save the density map in 
    DenMap_dir = os.path.join(root,'ShanghaiTech/part_A/train_data/Density_Map_GT')
    if not os.path.exists(DenMap_dir):
        os.mkdir(DenMap_dir)

    img_paths = []
    for img_path in glob.glob(os.path.join(part_A_train, "*.jpg")):
        img_paths.append(img_path)
      
    for img_path in img_paths:
        print(img_path)
        GT_path = img_path.replace('.jpg','.mat').replace('images','ground-truth').replace('IMG_','GT_IMG_')
        
        #load labelled data 
        mat = loadmat(GT_path)
        img = plt.imread(img_path)
        k = np.zeros((img.shape[0],img.shape[1]))
        gt = mat["image_info"][0,0][0,0][0] #2D locations of where the heads are 
        
        #convert the tuple of 2D locations to 2-D matrix of 0's and 1's 
        #to feed into the find density function
        for i in range(0,len(gt)):
            k[int(gt[i][1]),int(gt[i][0])]=1    

        k = gaussian_filter_density(k)
        #print(np.sum(k))
        
        #save the density map
        DenMap_save_path = img_path.replace('.jpg','.h5').replace('images','Density_Map_GT')
        with h5py.File(DenMap_save_path, 'w') as hf:
            hf['density'] = k
       


if __name__ == '__main__':

    Generate_Density_Map_Batch()    #generate the density map for each image
    Generate_Crop_Density()         #generate the crop images and the corresponding density map
    #Read_Density_Map('IMG_4_1.h5')  #sanity check  
   















