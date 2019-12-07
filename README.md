# Multi_Columns_Crowd_Counting

# Data Setup 
Download ShanghaiTech Dataset from
https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0

Put the folder ShanghaiTech in the same folder as the code files.

Run python3 Generate_Density_map.py 

Manually, move all the files in ./ShanghaiTech/part_A/train_data/images to ./ShanghaiTech/part_A/train_data/images_crop

Our data_loader.py code will read images and corresponded density map from there

Then, to create validation set, copy the folder ./ShanghaiTech/part_A/train_data and rename the copied folder to ShanghaiTech/part_A/val_data. For ./ShanghaiTech/part_A/train_data, delete the last 10% of the images. For ./ShanghaiTech/part_A/val_data, keep the last 10% of the images. 

# Train Each Column 
Run python3 train_col1.py to train column 1.
Run python3 train_col2.py to train column 2.
Run python3 train_col3.py to train column 3.

# Train the Fuse 
In trian_fuse.py, change the name of the paths of pre-trained models to ones you want to load. 
Then run trian_fuse.py to train the fusing process of the three columns.

# Testing Process
In test_each_column.py, make sure to change the paths of pre-trained models to the correct name. 
Run python3 test_each_column.py to get the result. 

Similarly, in test_combined_model.py, make sure to change the paths of pre-trained models to the correct name. Then, 
run python3 test_combined_model.py to get the result. 






