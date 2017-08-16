import os
import cv2
import argparse
import pandas
import json 
import numpy as np
from tqdm import tqdm
from utils import central_peripheral_seperator

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, help="Video directory.")
    parser.add_argument("--output_dir", type=str, help="Output directory.")
    parser.add_argument("--skip_frames", type=int, help="number of frames to skip")
    parser.add_argument("--from_list", action="store_true", help="Whether to run video in the list or the whole video directory")
    parser.add_argument("--with_video", action="store_true", help="Whether to output optical flow video")
    parser.add_argument("--with_json", action="store_true", help="Whether to output the json file")
    parser.add_argument("--sep_FOV", type=float, help="angle that seperate central and peripheral flow")
    parser.set_defaults(from_list=False)
    parser.set_defaults(with_video=False)
    parser.set_defaults(with_json=False)
    
    return parser.parse_args()

def optic_video(video_path, output_dir, filename, skip_frames, sep_FOV, with_video=False, with_json=False):
    if not os.path.exists(output_dir):
        print("%s not exist, creating..." % output_dir)
        os.mkdir(output_dir)

    if os.path.exists(os.path.join(output_dir, filename + '_flow_data.json')):
        print('%s already procceeding, pass...' % os.path.join(output_dir, filename + '_flow_data.json'))
        return
        
    json_f = open(os.path.join(output_dir, filename + '_flow_data.json'), 'w')
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    sep = central_peripheral_seperator(sep_FOV, prvs.shape)
    
    if with_video:
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')     #works, large

        out = cv2.VideoWriter(os.path.join(output_dir, filename + "_optic.avi"), fourcc, fps, size, True)

    flow_data = []
    accu_flow = np.zeros((int(prvs.shape[0]), int(prvs.shape[1]), 2))
    try:
        for i in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame2 = cap.read()
        
            if type(frame2) == type(None):
                break

            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 60, 3, 5, 1.2, 0)
            accu_flow += flow

            if i % skip_frames == 0 and with_json:
                '''
                mag_total, ang_total = cv2.cartToPolar(np.array(np.mean(accu_flow[...,0])), np.array(np.mean(accu_flow[...,1])))
                mag = np.power(np.power(accu_flow[...,0], 2) + np.power(accu_flow[...,1], 2), 1.0 / 2)
                central_mag, peripheral_mag = sep.seperate(mag)
                '''
                central_x, peripheral_x = sep.seperate(accu_flow[..., 0])
                central_y, peripheral_y = sep.seperate(accu_flow[..., 1])

                central_mag = np.power(np.power(central_x, 2) + np.power(central_y, 2), 1.0 / 2)
                peripheral_mag = np.power(np.power(peripheral_x, 2) + np.power(peripheral_y, 2), 1.0 / 2)

                central_mag_sum, central_ang_sum = cv2.cartToPolar(np.array(np.mean(central_x)), np.array(np.mean(central_y)))
                peripheral_mag_sum, peripheral_ang_sum = cv2.cartToPolar(np.array(np.mean(peripheral_x)), np.array(np.mean(peripheral_y)))
                total_mag_sum, total_ang_sum = cv2.cartToPolar(np.array(np.mean(accu_flow[...,0])), np.array(np.mean(accu_flow[...,1])))

                #flow_data.append([float(np.mean(central_mag)),float(np.mean(peripheral_mag)), float(mag_[0][0]), float(ang_total[0][0])])
                flow_data.append([float(np.mean(central_mag)),
                                  float(np.mean(peripheral_mag)), 
                                  float(central_mag_sum[0][0]), 
                                  float(central_ang_sum[0][0]),
                                  float(peripheral_mag_sum[0][0]), 
                                  float(peripheral_ang_sum[0][0]),
                                  float(total_mag_sum[0][0]), 
                                  float(total_ang_sum[0][0])])

                accu_flow = np.zeros((int(prvs.shape[0]), int(prvs.shape[1]), 2))

            if with_video:
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                hsv[...,0] = ang*180/np.pi/2
                #hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                hsv[...,2] = np.clip(mag * 5, 0, 255)
                bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
                out.write(bgr)
    
            prvs = next
        

    except KeyboardInterrupt:
        cap.release()
        if with_video:
            out.release()
        cv2.destroyAllWindows()
        json.dump(flow_data, json_f, indent=4)

    cap.release()
    if with_video:
        out.release()
    cv2.destroyAllWindows()
    json.dump(flow_data, json_f, indent=4)

def get_SSQ(df, subject, mode):
    subject_list =  df.loc[1:18,'User study 2 SSQ']
    row = None
    for i in range(len(subject_list)):
        if subject == ''.join(subject_list[i + 1].split()):
            row = i + 1
    column = None
    if mode == 'Chair':
        column = 'Unnamed: 2'
    elif mode == 'Rope':
        column = 'Unnamed: 3'
    elif mode == 'JumpGlide':
        column = 'Unnamed: 4'

    if row == None or column == None:
        return -1
    else:
        return df.loc[row, column]

def from_list(args):
    if not os.path.exists(args.output_dir):
        print("%s not exist, creating..." % args.output_dir)
        os.mkdir(args.output_dir)

    video_list = []
    with open("video_list.txt", 'r') as f:
        for line in f:
            video_list.append(line.strip().split())

    for v in video_list:
        video_path = os.path.join(args.video_dir, v[0], v[1])
        output_dir = os.path.join(args.output_dir, v[0])
        filename = v[1][:-4]
        print video_path, output_dir, filename
        optic_video(video_path, output_dir, filename, args.skip_frames, with_video=args.with_video)

def from_dir(args):
    if not os.path.exists(args.output_dir):
        print("%s not exist, creating..." % args.output_dir)
        os.mkdir(args.output_dir)
        
    f = open(os.path.join(args.output_dir, 'video_list.txt'), 'w')
    df = pandas.read_csv('SSQ.csv')
    
    subjects = [o for o in os.listdir(args.video_dir) if os.path.isdir(os.path.join(args.video_dir,o))]
    for s in subjects:
        subject_dir = os.path.join(args.output_dir, s)
        if not os.path.exists(subject_dir):
            print("%s not exist, creating..." % subject_dir)
            os.mkdir(subject_dir)
        
        videos = [v for v in os.listdir(os.path.join(args.video_dir, s)) if v[-4:] == '.mp4']
        for v in videos:
            video_path = os.path.join(args.video_dir, s, v)
            output_dir = os.path.join(args.output_dir, s)
            filename = v[:-4]
            mode = filename.split('_')[0]
            if mode[-1] == '1' or mode[-1] == '2':
                mode = mode[:-1]
            if mode == 'Teleport':
                continue
            
            #print video_path, output_dir, filename
            
            SSQ = get_SSQ(df, s, mode)
            
            optic_video(video_path, output_dir, filename, args.skip_frames, args.sep_FOV, with_video=args.with_video, with_json=args.with_json)
            f.write(' '.join([s, v, str(SSQ)]) + '\n')

if __name__ == '__main__':
    args = get_args()
    if args.from_list:
        from_list(args)
    else:
        from_dir(args)







