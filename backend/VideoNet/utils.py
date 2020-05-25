import os
import cv2
import copy
import torch
import pickle
import random
import numpy as np
import subprocess
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional
from torch.autograd import Variable

# global path

KINETICS_FRAME_DIR = '/root/hmdb51_frames/'
TRAIN_LIST = '/root/VideoNet/segment/train.list'
VAL_LIST = '/root/VideoNet/segment/hmdb_val.list'
TEST_LIST = '/root/VideoNet/segment/hmdb_test.list'

# gloable variable
num_classes_train = 10
VIDEO_FRAMES = 4
IMG_INIT_H=256
IMG_crop_size = (224,224)


#gloable functions


# for Video Processing
class ClipRandomCrop(torchvision.transforms.RandomCrop):
  def __init__(self, size):
    self.size = size
    self.i = None
    self.j = None
    self.th = None
    self.tw = None

  def __call__(self, img):
    if self.i is None:
      self.i, self.j, self.th, self.tw = self.get_params(img, output_size=self.size)
      #print('crop:', self.i, self.j, self.th, self.tw)
    return torchvision.transforms.functional.crop(img, self.i, self.j, self.th, self.tw)

class ClipRandomHorizontalFlip(object):
  def __init__(self, ratio=0.5):
    self.is_flip = random.random() < ratio

  def __call__(self, img):
    if self.is_flip:
      return torchvision.transforms.functional.hflip(img)
    else:
      return img

def transforms(mode):
    if (mode=='train'):
        random_crop = ClipRandomCrop(IMG_crop_size)
        flip = ClipRandomHorizontalFlip(ratio=0.5)
        toTensor = torchvision.transforms.ToTensor()
        normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return torchvision.transforms.Compose([random_crop, flip,toTensor,normalize])
    else:   # mode=='test'
        center_crop = torchvision.transforms.CenterCrop(IMG_crop_size)
        toTensor = torchvision.transforms.ToTensor()
        normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return torchvision.transforms.Compose([center_crop,toTensor,normalize])





# the basic , only can be used when video_frames=16 , (frames_num>16, return video frames unequal
def get_video_from_video_info(video_info, video_frames= VIDEO_FRAMES,frame_dir=KINETICS_FRAME_DIR,mode='train',data_aug=None):
    '''
    :param video_info: air drumming/-VtLx-mcPds_000012_000022
    :param mode:  train{random continuous 32-clip random-crop} , test{middle continuous 32-clip center-crop}
    :modality='rgb' defaulgt; 'depth': depth-frame; 'depth-1d': 1d depth; 'depth-4d': rgb+1d depth
    :return: torch.Size([16, 3, 242, 242])
    '''

    video_frame_path = os.path.join(frame_dir,video_info)
    all_frame_count = len(os.listdir(video_frame_path))-2
  

    if(all_frame_count -video_frames-1 >1):
        if (mode == 'train'):
            image_start = random.randint(1, all_frame_count - video_frames -1)
        # get middle 32-frame clip
        elif ((mode == 'test') | (mode=='val')):
            image_start = all_frame_count // 2 -video_frames // 2 + 1
    else:
        image_start=1   # use 0 padding

    image_id = image_start
    myTransform = transforms(mode=mode)
    video=[]
    for i in range(video_frames):
        s = "%05d" % image_id
        image_name = 'image_' + s + '.jpg'
        image_path = os.path.join(video_frame_path, image_name)
        image = Image.open(image_path)

        if (image.size[0] < 224):
            image = image.resize((224, IMG_INIT_H), Image.ANTIALIAS)
        image = myTransform(image)

        video.append(image)
        
    

        image_id += 4
        if (image_id > all_frame_count):
            image_id = all_frame_count

    video=torch.stack(video,0)
    #print(video.size())
    
    # add image gaussian [0,10]
    mu,sigma = 0,0.3
    if (data_aug =='aug_image_gaussian'):
        video = video.numpy()
        video[:seg_len,:,:,:] = video[:seg_len,:,:,:] + np.random.normal(mu,sigma,size=(seg_len,video.shape[1],video.shape[2],video.shape[3]))
        video = torch.FloatTensor(video)
   
   
    return video



def get_classname_from_video_info(video_info):
    '''
    :param video_info: air drumming/-VtLx-mcPds_000012_000022
    :return: classnum:air drumming
    '''
    video_info_splits = video_info.split('/')
    class_num = video_info_splits[0]
    return class_num

# global functions
def get_classInd(info_list):
    info_list=open(info_list).readlines()
    classlabel=0
    classInd={}
    for info_line in info_list:
        info_line=info_line.strip('\n')
        videoname= get_classname_from_video_info(info_line)
        if videoname not in classInd.keys():
            classInd[videoname]=classlabel
            classlabel = classlabel +1
        else:
            pass
    return classInd

def get_label_from_video_info(video_info,info_list = TRAIN_LIST):
   classname = get_classname_from_video_info(video_info)
   classInd = get_classInd(info_list)
   label = classInd[classname]
   return label


def get_label_from_classId(id, info_list=TRAIN_LIST):
    info_list = open(info_list).readlines()
    video_info = info_list[id]
    video_info = video_info.strip('\n')
    for i in range(len(video_info)):
        if video_info[i] == '/':
            break
    return video_info[:i]
    
    
