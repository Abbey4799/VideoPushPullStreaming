import cv2
import os
import  numpy as np
import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from utils import *
from dataloader import VideoDataset
from model import *


class TrainNetwork():
    def __init__(self, ckp_path, epoch_nums, batch_size, lr, lr_step_size=10, resnet_model='resnet50', num_classes = num_classes_train, pre_model_rgb=None):
        # get params
        self.ckp_path = ckp_path
        self.epoch_nums = epoch_nums
        self.batch_size = batch_size
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.resnet_model = resnet_model
        self.num_classes = num_classes
        
        self.pre_model_rgb = pre_model_rgb

        # mkdir ckg_path
        if not os.path.exists(self.ckp_path):
            os.makedirs(self.ckp_path)

        # load model
        self.mymodel = model_rgb(batch_size = self.batch_size, num_classes=self.num_classes)
        self.mymodel.train()

        if(pre_model_rgb):
            checkpoints = torch.load(pre_model_rgb, map_location=lambda storage, loc: storage)
            self.mymodel.load_state_dict(checkpoints)
            print('submodel_rgb loaded from {}'.format(pre_model_rgb))
       
        #self.mymodel = torch.nn.DataParallel(self.mymodel)
        #self.mymodel.cuda()
        print('model loaded.')

        # define video_dataloader
        self.myDataset = VideoDataset(KINETICS_FRAME_DIR, TRAIN_LIST, mode='train')
        self.myDataloader = DataLoader(self.myDataset, batch_size= self.batch_size, shuffle=True, num_workers=0)

        self.testDataset = VideoDataset(KINETICS_FRAME_DIR,TEST_LIST, mode='test')
        self.testDataloader = DataLoader(self.testDataset, batch_size= self.batch_size, shuffle=False, num_workers=0)


    def finetune_model(self,L2= True ):

        # define params
        params_list = list(self.mymodel.convnet.parameters()) 
        params_list2 = list(self.mymodel.fc.parameters()) + list(self.mymodel.fc2.parameters())

        optimizer = optim.Adagrad(params_list, lr= self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = self.lr_step_size, gamma=0.1)

        optimizer2 = optim.SGD(params_list2, lr= self.lr*10)
        scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size = self.lr_step_size, gamma=0.1)

        
        criterion = nn.CrossEntropyLoss()
	
	    # training process
        for epoch in range(self.epoch_nums):
            running_loss=0.0
            total_loss = 0.0
            total_acc = 0.0
            for i_batch, sample_batched in enumerate(self.myDataloader):
                # get the inputs
                video, label=sample_batched['video'], sample_batched['label']
                label= label.view((label.shape[0])).type(torch.LongTensor)
                label = Variable(label)#.cuda()
                #print(video.size())
                video = video.view(-1, video.size(2), video.size(3), video.size(4))
                #print(video.size())
                # zero the parameter gradients
                optimizer.zero_grad()
                optimizer2.zero_grad()
               
                
                # forward   
                video = Variable(video)#.cuda()
                output =self.mymodel(video)
                #print(output.size())
                loss = criterion(output,label)
                loss.backward()
                optimizer.step()
                optimizer2.step()
               
                
                label = label.data.cpu().numpy()
                output = output.data.cpu().numpy()
                predicted_y = np.argmax(output, axis=1)
                accuracy = np.mean(label == predicted_y)

                # print statistics
                running_loss = loss.item()
                total_loss += running_loss
                total_acc += accuracy
                
                print('training. [%d, %5d] loss: %.3f accuracy: %.3f' %(epoch + 1, i_batch + 1, running_loss, accuracy))
                

            scheduler.step()
            scheduler2.step()
            print('Epoch ',epoch,' avg loss: ',total_loss/15, ' total acc: ',total_acc/15)
            total_loss = 0.0
            total_acc = 0.0
            

            if (epoch+1)%50 == 0:
                save_model_path= self.ckp_path +'model'+str(epoch+1)+'.pkl'
                torch.save(self.mymodel.state_dict(),save_model_path)
                print('model saved.')
           
            '''
            testing_loss=0.0
            test_acc = 0.0 
            for i_batch, sample_batched in enumerate(self.testDataloader):
                # get the inputs
                video, label=sample_batched['video'], sample_batched['label']
                label= label.view((label.shape[0])).type(torch.LongTensor)
                label = Variable(label)#.cuda()

                video = video.view(-1, video.size(2), video.size(3), video.size(4))
                
                  
                video = Variable(video)#.cuda()
                output =self.mymodel(video)
                
                loss2 = criterion(output,label)
                label = label.data.cpu().numpy()
                output = output.data.cpu().numpy()
                predicted_y = np.argmax(output, axis=1)
                accuracy2 = np.mean(label == predicted_y)

                # print statistics
                testing_loss += loss2.item()
                test_acc += accuracy2
                print('testing. [%d, %5d] loss: %.3f' %(epoch + 1, i_batch + 1, loss2.item()))

            print('Epoch ',epoch,' avg loss: ',testing_loss/5, ' total acc: ',total_acc/5)
            print('testing completed.')
            testing_loss = 0.0
            test_acc = 0.0
            '''            

            


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    
    
    ckp_path, epoch_nums,batch_size,lr, lr_step_size, resnet_model = './result/model_rgb_1_batchsize/', 300, 1, 0.00005, 100, 'resnet50'
    pre_model_rgb = None
    
    myTrainNetwork = TrainNetwork(ckp_path,epoch_nums,batch_size,lr, lr_step_size,resnet_model,pre_model_rgb=pre_model_rgb)
    myTrainNetwork.finetune_model()

    



