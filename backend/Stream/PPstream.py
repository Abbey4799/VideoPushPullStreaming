import cv2
import subprocess
import numpy as np
from PIL import Image
import paddlehub as hub
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import argparse

 
 
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

# 创建解析器
parser = argparse.ArgumentParser() 

#添加位置参数(positional arguments)
parser.add_argument('--video_info', type=str, help='input the name of the video file')
args = parser.parse_args()
video_info = args.video_info


frame_dir = '/root/VideoNet/video'
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
        result = pose_estimation.keypoint_detection(images=images, batch_size=1,use_gpu=True, output_dir='/root/output_pose/'+str(image_cnt), visualization=True)
        images = os.listdir('/root/output_pose/'+str(image_cnt))
        image = os.path.join('/root/output_pose',images[0])
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
