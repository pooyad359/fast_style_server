import argparse
import time
import cv2
import torch
from torchvision import transforms
import utils
import re
from transformer_net import TransformerNet
import pathlib
from imutils.video import VideoStream
from imutils import resize
import numpy as np
import os
import itertools

device=torch.device('cpu')
parser=argparse.ArgumentParser()

parser.add_argument('--model', default='./models', help='Path to style model.')
parser.add_argument('--width', default=320, type=float,
                    help='For scaling the image. Default is 1 which keeps the image unchanged.')
parser.add_argument('--gpu',default=0,type=int,
                    help='If it is non-zero gpu will be used for inference.')
parser.add_argument('--time',default=None,type=float,
                    help='To activate automatic looping through the model set how long each model should be active.')
parser.add_argument('--full-screen', default = 0, type = int,
                    help = 'Display the output in full screen mode.')
parser.add_argument('--half',default=0,type=int,
                    help = 'If "0" uses float32, if "1" uses float16.')
parser.add_argument('--rotate',default = 0, type = int, 
                    help = 'if "1" will rotate the image 90 degrees CW, if "-1" will rotate the image 90 degrees CCW')
class Timer():
    def __init__(self):
        self.end = time.time()
        self.start=self.end

    def __call__(self,show=True):
        self.start = self.end
        self.end = time.time()
        if show:
            print(f'dt = {self.time():.3},\t FPS = {self.fps():.3}')

    def time(self):
        return self.end-self.start

    def fps(self):
        return 1/self.time()
    

def style_frame(img,style_model,device=device,half_precision=False):
    
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(img)
    content_image = content_image.unsqueeze(0).to(device)
    if half_precision:
        content_image = content_image.half()
    with torch.no_grad():
        output = style_model(content_image).cpu()
    output = output[0].numpy()
    output=np.clip(output,0,255)
    output=output.astype(np.uint8)
    output = output.transpose(1, 2, 0)
    return output


def multi_style(path,width=320,device=device,cycle_length = np.inf,half_precision=False, rotate = 0):
    model_iter=itertools.cycle(os.listdir(path))
    model_file=next(model_iter)
    print(f'Using {model_file} ')
    model_path=os.path.join(path,model_file)
    model = TransformerNet()
    model.load_state_dict(read_state_dict(model_path))
    model.to(device)
    if half_precision:
        model.half()
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    timer=Timer()
    cycle_begin = time.time()
    while(True):
        frame=vs.read()
        if frame is None:
            frame=np.random.randint(0,255,(int(width/1.5),width,3),dtype=np.uint8)
        frame=cv2.flip(frame, 1)
        frame = resize(frame, width=width)

        # Style the frame
        img=style_frame(frame,model,device,half_precision)
        img=cv2.resize(img[:,:,::-1],(640,480))

        # rotate
        if rotate>0:
            img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
        elif rotate<0:
            img = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
        # print(img.shape)
        cv2.imshow("Output", img)
        timer()
        key = cv2.waitKey(1) & 0xFF
        if key == ord("n") or (time.time()-cycle_begin)>cycle_length:
            model_file=next(model_iter)
            print(f'Using {model_file} ')
            model_path=os.path.join(path,model_file)
            model.load_state_dict(read_state_dict(model_path))
            model.to(device)
            cycle_begin=time.time()
        elif key == ord("q"):
            break



def read_state_dict(path):
    state_dict=torch.load(path)
    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]
    return state_dict



if __name__=='__main__':
    # Parse input arguments
    args=parser.parse_args()
    gpu=args.gpu
    half_precision = (args.half==1)
    cycle = args.time
    rotate = args.rotate
    if cycle is None:
        cycle = np.inf
    if gpu!=0 and torch.cuda.is_available():
        device=torch.device('cuda')
    assert half_precision is False or device==torch.device('cuda'), "Half precision is not supported on CPU."
    print('Using {}'.format(device))
    path= pathlib.Path(args.model)
    width=np.int(args.width)
    if args.full_screen:
        cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Output",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    if path.is_file():
        # load model
        model = TransformerNet()
        # state_dict = torch.load(args.model)
        # for k in list(state_dict.keys()):
        #     if re.search(r'in\d+\.running_(mean|var)$', k):
        #         del state_dict[k]
        state_dict=read_state_dict(args.model)
        model.load_state_dict(state_dict)
        model.to(device)
        style_cam(style_model = model,
                    width = width,
                    half_precision = half_precision,
                    rotate = rotate)
    else:
        multi_style( path = path,
                    width = width,
                    device = device,
                    cycle_length = cycle,
                    half_precision = half_precision,
                    rotate = rotate)

