import pymongo
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
from demo import *

import subprocess
import paddlehub as hub
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


 
 
def frame2video(im_dir,video_dir,fps):
 
    im_list = os.listdir(im_dir)
    im_list.sort(key=lambda x: int(x.replace("frame","").split('.')[0]))  #最好再看看图片顺序对不
    img = Image.open(os.path.join(im_dir,im_list[0]))
    img_size = img.size #获得图片分辨率，im_dir文件夹下的图片分辨率需要一致
 
 
    # fourcc = cv2.cv.CV_FOURCC('M','J','P','G') #opencv版本是2
    fourcc = cv2.VideoWriter_fourcc(*'XVID') #opencv版本是3
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
    # count = 1
    for i in im_list:
        im_name = os.path.join(im_dir+i)
        frame = cv2.imdecode(np.fromfile(im_name, dtype=np.uint8), -1)
        videoWriter.write(frame)
        # count+=1
        # if (count == 200):
        #     print(im_name)
        #     break
    videoWriter.release()
    print('Images converted to video: ',video_dir)


def ppstream(video_info):
    pose_estimation = hub.Module(name="human_pose_estimation_resnet50_mpii")
    #human_parser = hub.Module(name="ace2p")
    #stylepro_artistic = hub.Module(name="stylepro_artistic")

    rtsp = "rtmp://aerber.cn/live/ykyliveteststream"
    rtmp = 'rtmp://96941.livepush.myqcloud.com/live/lalala?txSecret=187cfff3247f32c39576ca7bef85d62c&txTime=5EC15F7F'

    # 读取视频并获取属性
    cap = cv2.VideoCapture(rtsp)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    sizeStr = str(size[0]) + 'x' + str(size[1])

    command = ['ffmpeg',
        '-y', '-an',
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', sizeStr,
        '-r', '25',
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
        '-f', 'flv',
        rtmp]

    pipe = subprocess.Popen(command
    , shell=False
    , stdin=subprocess.PIPE
    )

    
    frame_dir = '/home/ubuntu/html/backend/VideoNet/video'
    video_path = os.path.join(frame_dir, video_info)

    image_cnt = 1
    while cap.isOpened():
        success,frame = cap.read()
        #print(type(frame))
        if success:
            images = [frame]
            s = "%05d"%image_cnt
            frame_path = 'image_' + s + '.jpg'
            frame_abs_path = os.path.join(video_path,frame_path)
            frame.save(frame_abs_path)
            result = pose_estimation.keypoint_detection(images=images, batch_size=1,use_gpu=True, output_dir='/home/output_pose/'+str(image_cnt), visualization=True)
            images = os.listdir('/home/output_pose/'+str(image_cnt))
            image = os.path.join('/home/output_pose',images[0])
            frame = mpimg.imread(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break    
            pipe.stdin.write(frame.tostring())

    cap.release()
    pipe.terminate()

    im_dir = video_path#帧存放路径
    video_dir = os.path.join(im_dir,'test.video') #合成视频存放的路径
    fps = 5 #帧率，每秒钟帧数越多，所显示的动作就会越流畅
    frame2video(im_dir, video_dir, fps)


class VideoDB(object):
    def __init__(self):
        myclient = pymongo.MongoClient('mongodb://localhost:27017/')
        mydb = myclient['project']
        collection = mydb['video']
        self.client = myclient
        self.collection = collection
 
    def insert_video(self, video_path, video_class):
        # 插入数据
        self.collection.insert_one(
            {"address": video_path, "class": video_class}
        )
        # 关闭连接
        self.client.close()
        print("Successfully insert ",video_path," to ",video_class)

    def query_video(self, video_class):
        # 数据查询
        myquery = { "class": video_class }
        mydoc = self.collection.find(myquery)
        print(video_class,' has the following videos:')
        addresses = []
        for x in mydoc:
            print(x['address'])
            addresses.append(x['address'])
        # 关闭连接
        self.client.close()
        return addresses
        
        
    def delete_video(self, video_path):
        # 删除数据
        myquery = { "address": video_path }
        self.collection.delete_one(myquery)
        # 关闭连接
        self.client.close()
        print("Successfully delete ",video_path)

    def update_video(self, video_path, new_path):
        # 修改数据
        myquery = { "address": video_path }
        newvalues = { "$set": { "address": new_path} }
        self.collection.update_one(myquery, newvalues)
        print("Successfully set ", video_path, " to ", new_path)
        # 关闭连接
        self.client.close()



def run(option, video_info, video_class)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    # 创建解析器
    #parser = argparse.ArgumentParser() 

    #添加位置参数(positional arguments)
    #parser.add_argument('--option',type=int, help='1 for classify, 2 for query')
    #parser.add_argument('--video_info', type=str, help='input the name of the video file', default='None')
    #parser.add_argument('--video_class', type=str, help='input the class you wanna query', default='None')
    
    #args = parser.parse_args()
    #option = args.option
    #video_info = args.video_info
    #video_class = args.video_class

    if option == 1:
        ppstream(video_info)
        pre_model_rgb = './result/model_rgb_classical_learning/model350.pkl'
        frame_dir = '/home/ubuntu/html/backend/VideoNet/video'
        myTestNetwork = TestNetwork(frame_dir=frame_dir,video_info=video_info, pre_model_rgb=pre_model_rgb)
        label = myTestNetwork.test_model()
        videodb = VideoDB()
        video_path = os.path.join(frame_dir, video_info)
        video_path = os.path.join(video_path,'test.mp4')
        videodb.insert_video(video_path, label)
        return label
        #videos = videodb.query_video(label)
        #videodb.delete_video("/root/VideoNet/video/firstvideo")
    else:
        videodb = VideoDB()
        videos = videodb.query_video(video_class)
        return videos



option = int(sys.argv[0])
video_info = sys.argv[1]
video_class = sys.argv[2]
run(option. video_info, video_class)


