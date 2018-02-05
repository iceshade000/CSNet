import torch
import torch.nn as nn
import math
#from math import round
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    outchannel_ratio = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)        
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.bn1(x)
        out = self.conv1(out)        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
       
        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(torch.zeros(batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]).cuda())               
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut 

        return out

class contribute(nn.Module):
    def __init__(self,inplanes,planes,stride=1):
        super(contribute,self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)

        #i = int(planes*3/4)
        i = planes
        self.conv12 = nn.Conv2d(inplanes, i, kernel_size=1, bias=False)
        self.bn12 = nn.BatchNorm2d(i)
        self.conv22 = nn.Conv2d(i, i, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn22 = nn.BatchNorm2d(i)
        self.conv32 = nn.Conv2d(i, planes * 4, kernel_size=1, bias=False)

        self.bn32 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        '''

        D = planes
        C = cardinate
        self.conv12 = nn.Conv2d(inplanes, D * C, kernel_size=1, bias=False)
        self.bn12 = nn.BatchNorm2d(D * C)
        self.conv22 = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn22 = nn.BatchNorm2d(D * C)
        self.conv32 = nn.Conv2d(D * C, planes * 4, kernel_size=1, bias=False)
        self.bn32 = nn.BatchNorm2d(planes * 4)
        '''


    def forward(self, x):
        #rou = self.relu(self.bn12(self.conv12(x)))
        rou = self.relu(self.bn12(self.conv12(self.bn1(x))))
        rou = self.relu(self.bn22(self.conv22(rou)))
        rou = self.bn32(self.conv32(rou))

        rou = F.avg_pool2d(rou, rou.size(3))
        return rou


class Bottleneck(nn.Module):
    outchannel_ratio = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.stride = stride
        stride=1

        self.con=contribute(inplanes,planes,stride)

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.outchannel_ratio, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes * Bottleneck.outchannel_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample




    def forward(self, x):
        if self.stride>1:
            x=F.avg_pool2d(x,2)
        rou = self.con(x)
        out = self.bn1(x)
        #rou = self.con(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
 
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out = self.bn4(out)

        temp = F.avg_pool2d(out, out.size(3))
        out = out - temp+rou

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(torch.zeros(batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]).cuda())               
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut 

        return out


class PyramidNet(nn.Module):

    def __init__(self, depth, alpha, num_classes, bottleneck=False):
        super(PyramidNet, self).__init__()   	
        self.inplanes = 16

        n = (depth - 2) /6
        if bottleneck == True:
            n = int(n * 2 / 3)
            block = Bottleneck
        else:
            block = BasicBlock
            
        self.addrate =alpha / (3*n*1.0)

        self.input_featuremap_dim = self.inplanes
        self.conv1 = nn.Conv2d(3, self.input_featuremap_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)

        self.featuremap_dim = self.input_featuremap_dim 
        self.layer1 = self.pyramidal_make_layer(block, n)
        self.layer2 = self.pyramidal_make_layer(block, n, stride=2)
        self.layer3 = self.pyramidal_make_layer(block, n, stride=2)

        self.final_featuremap_dim = self.input_featuremap_dim
        self.bn_final= nn.BatchNorm2d(self.final_featuremap_dim)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.final_featuremap_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def pyramidal_make_layer(self, block, block_depth, stride=1):
        downsample = None
        #if stride != 1: # or self.inplanes != int(round(featuremap_dim_1st)) * block.outchannel_ratio:
        #    downsample = nn.AvgPool2d((2,2), stride = (2, 2))

        layers = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        layers.append(block(self.input_featuremap_dim, int(round(self.featuremap_dim)), stride, downsample))
        for i in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            layers.append(block(int(round(self.featuremap_dim)) * block.outchannel_ratio, int(round(temp_featuremap_dim)), 1))
            self.featuremap_dim  = temp_featuremap_dim
        self.input_featuremap_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        #temp = F.avg_pool2d(x, x.size(3))
        #x = x - temp

        x = self.bn1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn_final(x)
        x = self.relu_final(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
