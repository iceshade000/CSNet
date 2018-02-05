'''
resnet for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class transfer(nn.Module):
    def __init__(self,inplanes, planes,cardinate,stride=1):
        super(transfer,self).__init__()
        D = planes
        C = cardinate
        self.conv1 = nn.Conv2d(inplanes, D * C, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D * C)
        self.conv2 = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D * C)
        self.conv3 = nn.Conv2d(D * C, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        temp = F.avg_pool2d(out, out.size(3))
        out=out-temp

        return out

class contribute(nn.Module):
    def __init__(self,inplanes,planes,cardinate=1,stride=1):
        super(contribute,self).__init__()


        #i = int(planes*3/4)
        i = planes
        self.bn=nn.BatchNorm2d(inplanes)
        self.conv12 = nn.Conv2d(inplanes, i, kernel_size=1, bias=False)
        self.bn12 = nn.BatchNorm2d(i)
        self.conv22 = nn.Conv2d(i, i, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn22 = nn.BatchNorm2d(i)
        self.conv32 = nn.Conv2d(i, planes * 4, kernel_size=1, bias=False)

        self.bn32 = nn.BatchNorm2d(planes * 4)
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
        rou = F.relu(self.bn12(self.conv12(self.bn(x))))
        rou = F.relu(self.bn22(self.conv22(rou)))

        rou = self.bn32(self.conv32(rou))
        rou = F.avg_pool2d(rou, rou.size(3))
        return rou


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes,stride=1,downsample=None):
        super(Bottleneck, self).__init__()

        C = 1
        self.tran=transfer(inplanes,planes,C)
        #self.bn=nn.BatchNorm2d(inplanes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.contri=contribute(inplanes,planes)


    def forward(self, x):
        if self.stride>1:
            x=F.avg_pool2d(x,self.stride)
        residual = x

        #x=self.bn(x)
        out = self.tran(x)
        rou = self.contri(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out =  out +rou+residual

        return out


class ResBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes,stride=1,downsample=None):
        super(ResBottleneck, self).__init__()

        C = 1
        self.tran=transfer(inplanes,planes,C)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride


    def forward(self, x):
        if self.stride>1:
            x=F.avg_pool2d(x,self.stride)
        residual = x

        out = self.tran(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out =  out +residual

        out = self.relu(out)
        return out


class UBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes,stride=1,downsample=None):
        super(UBottleneck, self).__init__()

        C = 1
        self.tran=transfer(inplanes,planes,C)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.contri=contribute(inplanes,planes)
        if stride>1:
            #self.proj = nn.Conv2d(inplanes, planes*4, kernel_size=3,stride=stride,  padding=1, bias=False)
            self.mean=nn.Conv2d(inplanes, planes*4, kernel_size=1, bias=False)
            self.proj=transfer(inplanes,planes,1,stride)
            self.bn=nn.BatchNorm2d(planes*4)
            #self.mean=contribute(inplanes,planes,stride)



    def forward(self, x):
        residual = x
        if self.stride>1:
            proj=self.proj(x)
            mean=F.avg_pool2d(self.bn(self.mean(x)),x.size(3))
            residual=proj+mean
            x=F.avg_pool2d(x,self.stride)


        out = self.tran(x)
        rou = self.contri(x)


        out = residual + out +rou

        out = self.relu(out)
        return out

class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, N=64,num_classes=10):
        super(ResNet_Cifar, self).__init__()

        self.inplanes = N
        self.conv1 = nn.Conv2d(3, N, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(N)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, N, layers[0])
        self.layer2 = self._make_layer(block, N*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, N*4, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(N*4* block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample=None

        if self.inplanes != planes * block.expansion  :
            print(self.inplanes)
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride,downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        #temp = F.avg_pool2d(x, x.size(3))
        #x=x-temp

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def basenet(N=3,width=64,**kwargs):
    model = ResNet_Cifar(Bottleneck, [N, N, N],N=width, **kwargs)
    return model




