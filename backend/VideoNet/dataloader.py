import torch
from torch.utils.data import Dataset, DataLoader
from utils import *




# dataloader for train
class VideoDataset(Dataset):
    def __init__(self, root_dir, info_txt, mode='train'):
        # set params
        self.root_dir=root_dir
        self.info_txt = info_txt
        self.info_list=open(self.info_txt).readlines()
        self.mode = mode

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, idx):
        info_line=self.info_list[idx]
        video_info=info_line.strip('\n')
        
        video = get_video_from_video_info(video_info, frame_dir = self.root_dir,mode=self.mode)
        video_label=get_label_from_video_info(video_info,self.info_txt)

        #sample = {'video': video, 'video_depth':video_depth, 'label': [int(video_label)]}
        sample = {'video': video,  'label': [int(video_label)]}

        sample['video'] = torch.FloatTensor(sample['video'])  
        sample['label'] = torch.FloatTensor(sample['label'])

        return sample
