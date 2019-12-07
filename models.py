import torch
import torch.nn as nn

#Fuse the three columns/branches
class MCNN(nn.Module):    
    def __init__(self, bn=False):
        super(MCNN, self).__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(1, 16, 9, stride=1 , padding=4),                    
                                     nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout2d(0),
                                     nn.MaxPool2d(2),
                                     
                                     nn.Conv2d(16, 32, 7, stride=1 , padding=3),     
                                     nn.ReLU(inplace=True),
                                     nn.Dropout2d(0),
                                     nn.MaxPool2d(2),

                                     nn.Conv2d(32, 16, 7, stride=1 , padding=3),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout2d(0),

                                     nn.Conv2d(16, 8, 7, stride=1 , padding=3),
                                     nn.ReLU(inplace=True),
                                     )

        self.branch2 = nn.Sequential(nn.Conv2d(1, 20, 7, stride=1 , padding=3),
                                     nn.BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout2d(0),
                                     nn.MaxPool2d(2),
                                     
                                     nn.Conv2d(20, 40, 5, stride=1 , padding=2),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout2d(0),
                                     nn.MaxPool2d(2),

                                     nn.Conv2d(40, 20, 5, stride=1 , padding=2),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout2d(0),

                                     nn.Conv2d(20, 10, 5, stride=1 , padding=2),
                                     nn.ReLU(inplace=True),
                                     )
 

        self.branch3 = nn.Sequential(nn.Conv2d(1, 24, 5, stride=1 , padding=2),
                             nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True),
                             nn.ReLU(inplace=True),
                             nn.Dropout2d(0),
                             nn.MaxPool2d(2),
                             
                             nn.Conv2d(24, 48, 3, stride=1 , padding=1),
                             nn.ReLU(inplace=True),
                             nn.Dropout2d(0),
                             nn.MaxPool2d(2),

                             nn.Conv2d(48, 24, 3, stride=1 , padding=1),
                             nn.ReLU(inplace=True),
                             nn.Dropout2d(0),

                             nn.Conv2d(24, 12, 3, stride=1 , padding=1),
                             nn.ReLU(inplace=True),
                             )

        #freeze the three columns by setting requires_grad = False for the weights and biases
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                m.weight.requires_grad = False
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
                    m.bias.requires_grad = False

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                m.weight.requires_grad = False
                nn.init.constant_(m.bias, 0)
                m.bias.requires_grad = False

        self.fuse = nn.Sequential( 
                              nn.Conv2d(30, 1, 1, stride=1),
                              nn.LeakyReLU(negative_slope=0.00001, inplace=False),
                                )

        #initialize the weight of the fuse part separately. requires_grad=True in this part
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, im_data):
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x1,x2,x3),1)
        x = self.fuse(x)
        return x


