import numpy as np
import cv2
import os
import random
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm as CM
import h5py

class ImageDataLoader():
    def __init__(self, data_path, gt_path, shuffle, pre_load):
        self.shuffle = shuffle

        self.data_path = data_path
        self.gt_path = gt_path
        self.pre_load = pre_load

        self.data_files = []
        for filename in os.listdir(data_path):
            self.data_files.append(filename)
    
        if '.DS_Store' in self.data_files:
            self.data_files.remove('.DS_Store')
        
        self.num_samples = len(self.data_files)
        self.id_list = [*range(0,self.num_samples)]
        self.blob_list = {}        
        
        if self.pre_load:
            idx = 0
            for filename in self.data_files:
                img = cv2.imread(os.path.join(self.data_path, filename),0)
                ht = img.shape[0]
                wd = img.shape[1]
                
                img = img.reshape((1, 1, img.shape[0], img.shape[1]))
                
                
                den_path = os.path.join(self.gt_path,os.path.splitext(filename)[0] + '.h5')
                with h5py.File(den_path, 'r') as hf:
                    den = hf['density'][:]
                
                #down sample the gt density
                wd_1 = int(wd/4)
                ht_1 = int(ht/4)
                den = cv2.resize(den,(wd_1,ht_1))                
                den = den * ((wd*ht)/(wd_1*ht_1))

                #reshape to load into the model
                den = den.reshape((1,1,den.shape[0],den.shape[1]))            
                
                blob = {}
                blob['data']=img
                blob['gt_density']=den
                blob['filename'] = filename
                self.blob_list[idx] = blob
                idx = idx+1

                if idx % 100 == 0:                    
                    print ('Loaded ', idx, '/', self.num_samples, 'files')
            print ('Completed Loading ', idx, 'files')

    #generating function
    def __iter__(self):
        if self.shuffle:            
            if self.pre_load:            
                random.shuffle(self.id_list)          
            else:
                random.shuffle(self.data_files)       
        data_files = self.data_files
        id_list = self.id_list
       
        for idx in id_list:      
            if self.pre_load:
                blob = self.blob_list[idx]    
                blob['idx'] = idx 
            else:       
                filename = data_files[idx]
                img = cv2.imread(os.path.join(self.data_path, filename),0)
                            
                ht = img.shape[0]
                wd = img.shape[1] 
                img = img.reshape((1, 1, img.shape[0], img.shape[1]))
                            
                den_path = os.path.join(self.gt_path, os.path.splitext(filename)[0] + '.h5')
                with h5py.File(den_path, 'r') as hf:
                    den = hf['density'][:]
                
                #down sample the gt density
                wd_1 = int(wd/4)
                ht_1 = int(ht/4)
                den = cv2.resize(den,(wd_1,ht_1))                
                den = den * ((wd*ht)/(wd_1*ht_1))

                #reshape to load into the model
                den = den.reshape((1, 1, den.shape[0], den.shape[1]))            
                
                blob = {}
                blob['data']=img
                blob['gt_density']=den
                blob['filename'] = filename

            yield blob

            

    def get_num_samples(self):
        return self.num_samples
                
        
            
        










