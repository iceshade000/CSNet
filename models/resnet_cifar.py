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

moment=None
train = False

def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class SElayer(nn.Module):
    def __init__(self,inchannel,channel,reduction=16):
        super(SElayer,self).__init__()
        self.fc1=nn.Linear(inchannel,int(inchannel/reduction))
        self.fc2=nn.Linear(int(inchannel/reduction),channel)

    def forward(self,x):
        x1=torch.squeeze(F.avg_pool2d(x,x.size(3)))
        x1=F.relu(self.fc1(x1))
        x1=F.sigmoid(self.fc2(x1))
        return x1.view(-1,x1.size(1),1,1)



class SEBasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.se=SElayer(planes,planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)


        if self.downsample is not None:
            residual = self.downsample(x)
        a=self.se(out)
        out=out*a
        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion=4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.se = SElayer(inplanes,planes*4)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        #out = F.dropout2d(out, training=True)


        if self.downsample is not None:
            residual = self.downsample(x)

        #a = self.se(out)
        a=self.se(x)
        out = out * a

        out += residual
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)


        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class UBasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(UBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        i=int(planes/8)
        self.conv12 = conv3x3(inplanes, i, stride)
        self.bn12 = nn.BatchNorm2d(i)
        self.conv22 = conv3x3(i, planes)
        self.bn22 = nn.BatchNorm2d(planes)




    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)



        out = self.conv2(out)
        out = self.bn2(out)

        rou = F.relu(self.bn12(self.conv12(x)))
        rou = self.conv22(rou)
        #rou = F.relu(self.bn222(self.conv22(rou)))
        #rou =torch.squeeze(F.avg_pool2d(rou,rou.size(3)))
        #rou = self.fc(rou)
        rou = self.bn22(rou)
        #rou = rou.view(-1,rou.size(1),1,1)


        if self.downsample is not None:
            residual = self.downsample(x)

        rou=F.avg_pool2d(rou,rou.size(3))
        temp=F.avg_pool2d(out,out.size(3))

        #a=F.sigmoid(self.a)
        #mout = a*out+(1-a)*rou

        mout = out + rou-temp


        out =residual+mout
        out = self.relu(out)

        return out


class UBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes,stride=1, downsample=None):
        super(UBottleneck, self).__init__()

        D = planes
        C = 16
        self.conv1 = nn.Conv2d(inplanes, D * C, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D * C)
        self.conv2 = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D * C)
        self.conv3 = nn.Conv2d(D * C, planes *4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        '''
        i = planes
        self.conv12 = nn.Conv2d(inplanes, i, kernel_size=1, bias=False)
        self.bn12 = nn.BatchNorm2d(i)
        self.conv22 = nn.Conv2d(i, i, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn22 = nn.BatchNorm2d(i)
        self.conv32 = nn.Conv2d(i,planes*4, kernel_size=1, bias=False)
        #self.fc = nn.Linear(i,planes*4)

        self.bn32 = nn.BatchNorm2d(planes * 4)
        #self.bn32 = nn.BatchNorm1d(planes * 4)
        '''




    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if residual.size() != out.size():
            print(out.size(), residual.size())


        '''
        rou = F.relu(self.bn12(self.conv12(x)))
        rou = F.relu(self.bn22(self.conv22(rou)))

        rou = self.bn32(self.conv32(rou))
        rou = F.avg_pool2d(rou, rou.size(3))
        #rou = torch.squeeze(F.avg_pool2d(rou,rou.size(3)))

        #rou = self.bn32(self.fc(rou))
        #rou = rou.view(-1, rou.size(1), 1, 1)

        temp = F.avg_pool2d(out, out.size(3))
        mout = out + rou - temp
        out = residual + mout
        '''



        out += residual
        out = self.relu(out)
        return out


class UPABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(UPABasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

        self.conv12 = conv3x3(inplanes, planes, stride)
        self.bn12 = nn.BatchNorm2d(planes)
        self.conv22 = conv3x3(planes, planes)
        self.a = Parameter(torch.Tensor(1, planes, 1, 1))

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        rou = F.relu(self.bn12(self.conv12(out)))



        if self.downsample is not None:
            residual = self.downsample(out)


        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        rou = self.conv22(rou)

        rou = F.avg_pool2d(rou, rou.size(3))
        a = F.sigmoid(self.a)

        mout = a * (out - F.avg_pool2d(out, out.size(3))) + (1 - a) * rou

        out = residual + mout

        return out

class Bottleneck(nn.Module):
    expansion=4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        '''
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        '''

        D = planes
        C = 1
        self.conv1 = nn.Conv2d(inplanes, D * C, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D * C)
        self.conv2 = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D * C)
        self.conv3 = nn.Conv2d(D * C, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        #out = F.dropout2d(out, training=True)


        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        global moment
        a = 0.5

        if self.downsample is not None:
            residual = self.downsample(out)
            moment = None

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)


        if moment is None:
            moment = out
        else:
            moment = moment * a + out

        out = residual + moment

        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out


class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet_Cifar, self).__init__()
        N =16
        self.inplanes = N
        self.conv1 = nn.Conv2d(3, N, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(N)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, N, layers[0])
        self.layer2 = self._make_layer(block, N*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, N*4, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(N*4 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class PreAct_ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(PreAct_ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn = nn.BatchNorm2d(64*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        global moment
        moment = None
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def resnet20_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [3, 3, 3], **kwargs)
    return model

def resnet20_cifar_se(**kwargs):
    model = ResNet_Cifar(SEBasicBlock, [3, 3, 3], **kwargs)
    return model



def resnet32_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44_cifar(**kwargs):
    model = ResNet_Cifar(UBasicBlock, [7, 7, 7], **kwargs)
    return model

def preact_resnet44_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet1202_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [200, 200, 200], **kwargs)
    return model


def resnet164_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [18, 18, 18], **kwargs)
    return model


def resnet1001_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [111, 111, 111], **kwargs)
    return model


def preact_resnet110_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBasicBlock, [18, 18, 18], **kwargs)
    return model


def preact_resnet164_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [18, 18, 18], **kwargs)
    return model


def preact_resnet1001_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [111, 111, 111], **kwargs)
    return model


class Wide_ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, wfactor, num_classes=10):
        super(Wide_ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16*wfactor, layers[0])
        self.layer2 = self._make_layer(block, 32*wfactor, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64*wfactor, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64*block.expansion*wfactor, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def wide_resnet_cifar(depth, width, **kwargs):
    assert (depth - 2) % 6 == 0
    n = int((depth - 2) / 6)
    return Wide_ResNet_Cifar(BasicBlock, [n, n, n], width, **kwargs)

if __name__ == '__main__':
    net = preact_resnet110_cifar()
    y = net(torch.autograd.Variable(torch.randn(1, 3, 32, 32)))
    print(net)
    print(y.size())

