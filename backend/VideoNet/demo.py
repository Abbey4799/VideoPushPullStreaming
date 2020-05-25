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
import argparse

from utils import *
from model import *


class TestNetwork():
    def __init__(self, frame_dir, video_info, video_frames = 4, batch_size=2, num_classes = num_classes_train, pre_model_rgb=None):
        # get params
        self.frame_dir = frame_dir
        self.video_info = video_info
        self.video_frames = video_frames
        self.num_classes = num_classes
        self.pre_model_rgb = pre_model_rgb
        self.batch_size = batch_size

        # load model
        self.mymodel = model_rgb(batch_size = self.batch_size, num_classes=self.num_classes)
        self.mymodel.eval()

        if(pre_model_rgb):
            checkpoints = torch.load(pre_model_rgb, map_location=lambda storage, loc: storage)
            self.mymodel.load_state_dict(checkpoints)
            #print('submodel_rgb loaded from {}'.format(pre_model_rgb))
       
        #self.mymodel = torch.nn.DataParallel(self.mymodel)
        #self.mymodel.cuda()
        #print('model loaded.')

    def test_model(self):
        video = get_video_from_video_info(video_info=self.video_info, video_frames=self.video_frames,frame_dir=self.frame_dir, mode='test')
        #print(video.size())        
        #video = video.view(-1, video.size(2), video.size(3), video.size(4))
        video = Variable(video)#.cuda()
        video = torch.cat((video,video), dim=0)
        #print(video.size())
        output =self.mymodel(video)
        output = output.data.cpu().numpy()
        predicted_y = np.argmax(output, axis=1)
        #print('prediction: ',predicted_y)
        classid = predicted_y[0]
        label = get_label_from_classId(classid)
        print('The type of this video is: ',label)
        return label

                     

            


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    # 创建解析器
    parser = argparse.ArgumentParser() 

    #添加位置参数(positional arguments)
    parser.add_argument('--video_info', type=str, help='input the name of the video file')
    args = parser.parse_args()
    video_info = args.video_info
    pre_model_rgb = './result/model_rgb_classical_learning/model350.pkl'
    
    myTrainNetwork = TestNetwork(frame_dir='/root/VideoNet/video',video_info=video_info, pre_model_rgb=pre_model_rgb)
    label = myTrainNetwork.test_model()


    



