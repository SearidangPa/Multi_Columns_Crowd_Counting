import torch
import torch.nn as nn

#The second column of the multi-column CNN
class MCNN_col2(nn.Module):
    def __init__(self, bn=False):
        super(MCNN_col2, self).__init__()
        
        self.branch2 = nn.Sequential(nn.Conv2d(1, 20, 7, stride=1 , padding=3),
                                     nn.BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True),
                                     #inplace = True means that the model can modify the data in place. Save space.
                                     nn.ReLU(inplace=True),
                                     nn.Dropout2d(0.3),
                                     nn.MaxPool2d(2),
                                     
                                     nn.Conv2d(20, 40, 5, stride=1 , padding=2),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout2d(0.2),
                                     nn.MaxPool2d(2),

                                     nn.Conv2d(40, 20, 5, stride=1 , padding=2),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout2d(0.1),

                                     nn.Conv2d(20, 10, 5, stride=1 , padding=2),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout2d(0.1),
                                     nn.Conv2d(10, 1, 1,  stride=1)
                                     )
        #initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #use normal distribution with mean 0 and standard deviation 0.1
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias) #initalize the bias with 0's

    def forward(self, im_data):
        x = self.branch2(im_data)
        return x


