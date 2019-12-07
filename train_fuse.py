import os
import torch
import numpy as np
import sys
import math
import utils
from data_loader import ImageDataLoader
from torch.autograd import Variable
from matplotlib import pyplot as plt
from matplotlib import cm as CM
from models import MCNN
from model_col1 import MCNN_col1
from model_col2 import MCNN_col2
from model_col3 import MCNN_col3
from torch.utils.tensorboard import SummaryWriter
np.set_printoptions(threshold=sys.maxsize)

#paths
output_save_path = './trained_models/'
train_path = './ShanghaiTech/part_A/train_data/images_crop'
train_gt_path = './ShanghaiTech/part_A/train_data/Density_Map_GT_Crop'
val_path = './ShanghaiTech/part_A/val_data/images_crop'
val_gt_path = './ShanghaiTech/part_A/val_data/Density_Map_GT_Crop'

#tensorboard log runs
writer = SummaryWriter("./runs")

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
            mse += ((gt_count-et_count)*(gt_count-et_count))        

    num_samples = data_loader_val.get_num_samples()
    mae = mae/num_samples
    mse = np.sqrt(mse/num_samples)
    return mae, mse


# pre-trained models
pre_trained_col1_path = "./pre_trained_model_col1/MCNN_col1_part_A_226.pth"
pre_trained_col2_path = "./pre_trained_model_col2/MCNN_col2_part_A_190.pth"
pre_trained_col3_path = "./pre_trained_model_col3/MCNN_col3_part_A_266.pth"
model = MCNN()
loss_fn = torch.nn.MSELoss()
opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001) 
 
epoch = 0
running_loss = 0.0
count_step = 0 
loss_per_epoch = 0.0

continue_training = True
if continue_training == True:
    model_name = "trained_models/MCNN_part_A_12.pth"
    trained_model_path = os.path.join(os.getcwd(), model_name)
    checkpoint = torch.load(trained_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    opt.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
else: #load pre-trained models
    """
    print("........................before loading..................")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        print(model.state_dict()[param_tensor])
    """
    model_col1 = MCNN_col1()
    model_col2 = MCNN_col2()
    model_col3 = MCNN_col3()
    checkpoint1 = torch.load(pre_trained_col1_path)
    checkpoint2 = torch.load(pre_trained_col2_path)
    checkpoint3 = torch.load(pre_trained_col3_path)
    model_col1.load_state_dict(checkpoint1['state_dict'])
    model_col2.load_state_dict(checkpoint2['state_dict'])
    model_col3.load_state_dict(checkpoint3['state_dict'])

    
    for param_tensor in model_col1.state_dict():
        if param_tensor != 'branch1.15.weight' and param_tensor != 'branch1.15.bias':
            param_col1 = model_col1.state_dict()[param_tensor]
            model.state_dict()[param_tensor].copy_(param_col1)

    for param_tensor in model_col2.state_dict():
        if param_tensor != 'branch2.15.weight' and param_tensor != 'branch2.15.bias':
            param_col2 = model_col2.state_dict()[param_tensor]
            model.state_dict()[param_tensor].copy_(param_col2)


    for param_tensor in model_col3.state_dict():
        if param_tensor != 'branch3.15.weight' and param_tensor != 'branch3.15.bias':
            param_col3 = model_col3.state_dict()[param_tensor]
            model.state_dict()[param_tensor].copy_(param_col3)

    for name, parameter in model.named_parameters():
    	print(name)
    	print(parameter.requires_grad)

    """
    print("........................after loading..................")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        print(model.state_dict()[param_tensor])
    """




data_loader_train = ImageDataLoader(train_path, train_gt_path, shuffle=True, pre_load = True)
data_loader_val = ImageDataLoader(val_path, val_gt_path, shuffle=False, pre_load = True)

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
            #print(density_map)
            print("density_map_gt", np.sum(gt_data))
            print("estimated_den_map", np.sum(density_map))
            utils.save_results(im_data,gt_data,density_map, output_save_path)
            
            # log the running loss
            writer.add_scalar('training loss', running_loss / 300,
                                epoch * data_loader_train.get_num_samples()+ count_step)
            print("running_loss: ", running_loss)
            running_loss = 0.0
        
    if (epoch % 2 == 0):
        model.eval()      #enter evaluation model
        save_name = os.path.join(output_save_path, '{}_{}_{}.pth'.format("MCNN","part_A",epoch))
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
 

        
        
        
        


