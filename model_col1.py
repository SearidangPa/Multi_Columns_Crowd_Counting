import torch
import torch.nn as nn


class MCNN_col1(nn.Module):
    
    def __init__(self, bn=False):
        super(MCNN_col1, self).__init__()
        
        self.branch1 = nn.Sequential(nn.Conv2d(1, 16, 9, stride=1 , padding=4),                    
                                     nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True),
                                     #inplace = True means that the model can modify the data in place. Save space. 
                                     nn.ReLU(inplace=True), 
                                     nn.Dropout3d(0.3),
                                     nn.MaxPool2d(2),
                                     
                                     nn.Conv2d(16, 32, 7, stride=1 , padding=3),     
                                     nn.ReLU(inplace=True),
                                     nn.Dropout3d(0.2),
                                     nn.MaxPool2d(2),

                                     nn.Conv2d(32, 16, 7, stride=1 , padding=3),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout3d(0.1),

                                     nn.Conv2d(16, 8, 7, stride=1 , padding=3),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout3d(0.1),
                                     nn.Conv2d(8, 1, 1,  stride=1)
                                     )
        #initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #use normal distribution with mean 0 and standard deviation 0.1
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias) #initalize the bias with 0's

    def forward(self, im_data):
        x = self.branch1(im_data)
        return x


