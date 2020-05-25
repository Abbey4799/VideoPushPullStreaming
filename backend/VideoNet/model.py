import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

'''
- single model for different modality
- 1. rgb: model_resnet 50
- 2. depth: model_resnet 50
- 3. object: model_object
'''
class model_resnet18(nn.Module):
    def __init__(self,num_classes):
        super(model_resnet18,self).__init__()

        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]     # delete the last fc layer.
        self.convnet = nn.Sequential(*modules)
        self.fc = nn.Linear(512,num_classes)

    def forward(self,x):
        feature = self.convnet(x)
        feature = feature.view(x.size(0), -1)
        output = self.fc(feature)
        return feature,output

class model_resnet50(nn.Module):
    def __init__(self,num_classes):
        super(model_resnet50,self).__init__()

        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]     # delete the last fc layer.
        self.convnet = nn.Sequential(*modules)
        self.fc = nn.Linear(2048,num_classes)

    def forward(self,x):
        feature = self.convnet(x)
        feature = feature.view(x.size(0), -1)
        #output = self.fc(feature)
        return feature#,output

# directly modified from model_resnet50
class model_rgb(nn.Module):
    def __init__(self,batch_size, num_classes):
        super(model_rgb,self).__init__()
        self.batch_size = batch_size
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]     # delete the last fc layer.
        self.convnet = nn.Sequential(*modules)
        self.fc = nn.Linear(512,128)
        self.fc2 = nn.Linear(128, num_classes)
        #h0 = torch.zeros(2, 1, 1024)
        #c0 = torch.zeros(2, 1, 1024)
        #self.hidden = (h0, c0)
        #self.lstm = nn.LSTM(input_size=2048,hidden_size=1024,num_layers=2)

    def forward(self,x):
        #print(x.size())
        feature = self.convnet(x)
        #print(feature.size())
        feature = feature.view(self.batch_size, feature.size(0)//self.batch_size, -1)
        #print(feature.size())
        feature = torch.mean(feature, dim=1)
        #print(feature.size())
        #feature = feature.unsqueeze(0)
        output = self.fc(feature)
        output = self.fc2(output)
        #output = torch.nn.functional.softmax(output, dim=1)
        #print(output)
        #feature = feature.unsqueeze(1)
        #out,_ = self.lstm(feature, self.hidden)
        #out = out[0:1]
        #out = out.squeeze(1)
        #output = self.fc(out)
        #print(output.size())
        return output


def get_model(num_classes, pre_model_rgb=None):
    model = model_rgb(num_classes = num_classes)
    if(pre_model_rgb):
        model.load_state_dict(torch.load(pre_model_rgb))
        print('submodel_rgb loaded from {}'.format(pre_model_rgb))
   
    return model



