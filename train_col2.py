import os
import torch
import numpy as np
import sys
import math
from data_loader import ImageDataLoader
from torch.autograd import Variable
from matplotlib import pyplot as plt
from matplotlib import cm as CM
from model_col2 import MCNN_col2
from torch.utils.tensorboard import SummaryWriter

#paths 
output_save_path = './pre_trained_model_col2/'
train_path = './ShanghaiTech/part_A/train_data/images_crop'
train_gt_path = './ShanghaiTech/part_A/train_data/Density_Map_GT_Crop'
val_path = './ShanghaiTech/part_A/val_data/images_crop'
val_gt_path = './ShanghaiTech/part_A/val_data/Density_Map_GT_Crop'

#Tensorboard log runs 
writer = SummaryWriter("./runs_col2")

def evaluate_model(model, data_loader_val):
	print("validation evaluation")
	mae = 0.0
	mse = 0.0
	for blob in data_loader_val:                       
		im_data = blob['data']
		gt_data = blob['gt_density']
		gt_count = np.sum(gt_data)
		with torch.no_grad():
			im_data = torch.from_numpy(im_data).type(torch.FloatTensor)
			density_map = model(im_data)
			density_map = density_map.data.cpu().numpy()
			et_count = np.sum(density_map)
			mae += abs(gt_count-et_count)
			mse += mae*mae        

	num_samples = data_loader_val.get_num_samples()
	mae = mae/num_samples
	mse = np.sqrt(mse/num_samples)
	return mae, mse

# initialize an instance of the model
model = MCNN_col2()

loss_fn = torch.nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.00001)  
epoch = 0

continue_training = True
if continue_training == True:
	model_name = "pre_trained_model_col2/MCNN_col2_part_A_190.pth"
	trained_model_path = os.path.join(os.getcwd(), model_name)
	checkpoint = torch.load(trained_model_path)
	model.load_state_dict(checkpoint['state_dict'])
	opt.load_state_dict(checkpoint['optimizer'])
	epoch = checkpoint['epoch']

data_loader_train = ImageDataLoader(train_path, train_gt_path, shuffle=True, pre_load = True)
data_loader_val = ImageDataLoader(val_path, val_gt_path, shuffle=False, pre_load = True)




running_loss = 0.0
count_step = 0 
loss_per_epoch = 0.0

while epoch < 1000: 
	model.train()   #enter training mode 
	loss_per_epoch = 0.0
	print("epoch: ", epoch) 
	for blob in data_loader_train:                        
		im_data = blob['data']
		gt_data = blob['gt_density']

		im_data = torch.from_numpy(im_data).type(torch.FloatTensor)
		gt_data = torch.from_numpy(gt_data).type(torch.FloatTensor)

		# zero the parameter gradients
		opt.zero_grad()

		# forward + backward + optimize
		density_map = model(im_data)
		loss = loss_fn(density_map, gt_data)
		loss.backward()
		opt.step()

		running_loss += loss.item()
		loss_per_epoch += loss.item()
		count_step = count_step+1
		print(count_step)  
			
		if count_step % 300 == 0:    # every 300 mini-batches...
			density_map = density_map.data.cpu().numpy()
			gt_data = gt_data.data.cpu().numpy()
			print("density_map_gt", np.sum(gt_data))
			print("estimated_den_map", np.sum(density_map))
			
			# log the running loss
			writer.add_scalar('training loss', running_loss / 300,
							   epoch * data_loader_train.get_num_samples()+ count_step)
			print("running_loss: ", running_loss)	
			running_loss = 0.0
		

	if (epoch % 2 == 0):
		model.eval()      #enter evaluation model
		#save model every 2 epochs
		save_name = os.path.join(output_save_path, '{}_{}_{}.pth'.format("MCNN_col2","part_A",epoch))
		states = {
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'optimizer': opt.state_dict()
		}
		torch.save(states, save_name)

		#validation set evaluation
		mae,mse = evaluate_model(model, data_loader_val)
		writer.add_scalar('mae', mae , epoch) 
		writer.add_scalar('mse', mse , epoch) 
		print('mae', mae)


	writer.add_scalar('loss_per_epoch', loss_per_epoch , epoch) 
	print('loss_per_epoch: ', loss_per_epoch)
	epoch = epoch + 1

print('Finished Training')          
writer.close()
 
 
		
		
		
		


