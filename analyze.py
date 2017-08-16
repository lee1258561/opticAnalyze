import os
import cv2
import time
import argparse
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats import pearsonr
from utils import central_peripheral_seperator

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flow_dir', type=str, help="path to the flow data directory")
    parser.add_argument('--fig_dir', type=str, help="path to the figure directory")
    parser.add_argument('--filename', type=str, help="path to demo video file")
    parser.set_defaults(fig_dir=None) 

    return parser.parse_args()

def from_list(args):
    threshold = [8.0, 30.0, 45.0]
    video_list = []
    with open(os.path.join(args.flow_dir, "video_list.txt"), 'r') as f:
        for line in f:
            video_list.append(line.strip().split())

    subject_list, chair_subject, chair_data, rope_subject, rope_data, jump_subject, jump_data = [], [], [], [], [], [], []  
    data_list = {'central': [], 'peripheral': [], 'total': []}

    type_list = ['SSQ', 'mag_mean', 'mag', 'mag_sum', 'pulse_count', 'pulse_time', 'max_pulse_time', 'pulse_flow', 'max_pulse_flow', 'time']
    
    if args.fig_dir != None and not os.path.exists(args.fig_dir):
        os.mkdir(args.fig_dir)
    i = 0
    while i < len(video_list):
        v = video_list[i]
        print v
        mode = v[1].split('_')[0]

        
        flowData_path = os.path.join(args.flow_dir, v[0], v[1][:-4] + '_flow_data.json')
        flow_data = json.load(open(flowData_path, 'r'))
        if args.fig_dir != None:
            sub_dir = os.path.join(args.fig_dir, v[0])
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)
                
            fig_path = os.path.join(sub_dir, mode)
            draw(flow_data, fig_path, mode, v[2])
        
        mag, mag_mean, mag_sum, time = flow_analyze(flow_data)
        
        if mode[-1] == '1':
            i += 1
            v_part2 = video_list[i]
            mode_part2 = v_part2[1].split('_')[0]
            if v[0] != v_part2[0] or mode[:-1] != mode_part2[:-1] or mode_part2[-1] != '2':
                print('Error: part2 data mismatch')
                

            flowData_path = os.path.join(args.flow_dir, v_part2[0], v_part2[1][:-4] + '_flow_data.json')
            flow_data = json.load(open(flowData_path))
            mag2, mag_mean2, mag_sum2, time2 = flow_analyze(flow_data)
            
            
            mag = map(sum, zip(mag, mag2))
            mag_mean = np.average(zip(mag_mean, mag_mean2), axis=1, weights=[time, time2])
            mag_sum = map(sum, zip(mag_sum, mag_sum2))

            pulse_count, pulse_time, max_pulse_time, pulse_flow, max_pulse_flow = pulse_analyze(flow_data, mag_mean)
            pulse_count2, pulse_time2, max_pulse_time2, pulse_flow2, max_pulse_flow2 = pulse_analyze(flow_data, mag_mean)

            pulse_count = map(sum, zip(pulse_count, pulse_count2))
            pulse_time = map(sum, zip(pulse_time, pulse_time2))
            max_pulse_time = map(sum, zip(max_pulse_time, max_pulse_time2))
            pulse_flow = map(sum, zip(pulse_flow, pulse_flow2))
            max_pulse_flow = map(sum, zip(max_pulse_flow, max_pulse_flow2))
            time += time2

            mode = mode[:-1]
        else:
            pulse_count, pulse_time, max_pulse_time, pulse_flow, max_pulse_flow = pulse_analyze(flow_data, mag_mean)

        m, s = divmod(time * 15, 60)
        
        
        subject_list.append(v[0] + '_' + mode)
        data_list['central'].append([float(v[2]), mag_mean[0], mag[0], mag_sum[0], pulse_count[0], pulse_time[0], max_pulse_time[0], pulse_flow[0], max_pulse_flow[0],  ':'.join([str(m), str(s)])])
        data_list['peripheral'].append([float(v[2]), mag_mean[1], mag[1], mag_sum[1], pulse_count[1], pulse_time[1], max_pulse_time[1], pulse_flow[1], max_pulse_flow[1],  ':'.join([str(m), str(s)])])
        data_list['total'].append([float(v[2]), mag_mean[2], mag[2], mag_sum[2], pulse_count[2], pulse_time[2], max_pulse_time[2], pulse_flow[2], max_pulse_flow[2],  ':'.join([str(m), str(s)])])

        i += 1
    
    df = pd.DataFrame(np.array(data_list['central']), index=subject_list, columns=type_list)   
    df.to_csv(os.path.join(args.flow_dir, 'analysis_central.csv'))
    df = pd.DataFrame(np.array(data_list['peripheral']), index=subject_list, columns=type_list)   
    df.to_csv(os.path.join(args.flow_dir, 'analysis_peripheral.csv'))
    df = pd.DataFrame(np.array(data_list['total']), index=subject_list, columns=type_list)   
    df.to_csv(os.path.join(args.flow_dir, 'analysis_total.csv'))



def flow_analyze(flow_data):
    flow_data = np.array(flow_data)

    mag, mag_sum = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    for i in range(len(flow_data)):
        mag_list = [flow_data[i][0], flow_data[i][1], flow_data[i][0] + flow_data[i][1]]
        mag_sum_list = [flow_data[i][2], flow_data[i][4], flow_data[i][6]]
        #TODO: change to above
        #mag_sum_list = [0.0, 0.0, flow_data[i][2]]
        for j in range(len(mag_list)):
            mag[j] += mag_list[j]
            mag_sum[j] += mag_sum_list[j]

    time = len(flow_data) / 30
    mag_mean = [np.mean(flow_data[...,0]), np.mean(flow_data[...,1]), np.mean(flow_data[...,0]) + np.mean(flow_data[...,1])]

    return mag, mag_mean, mag_sum, time

def pulse_analyze(flow_data, threshold):
    pulse_count, pulse_time, max_pulse_time, pulse_flow, max_pulse_flow = [0, 0, 0], [0, 0, 0], [0, 0, 0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    tmp_time, tmp_flow = [0, 0, 0], [0.0, 0.0, 0.0]
    for i in range(len(flow_data)):
        mag_list = [flow_data[i][0], flow_data[i][1], flow_data[i][0] + flow_data[i][1]]
        for j in range(len(mag_list)):
            if mag_list[j] > threshold[j]:
                pulse_time[j] += 1
                tmp_time[j] += 1
                pulse_flow[j] += mag_list[j] - threshold[j]
                tmp_flow[j] += mag_list[j] - threshold[j]
            elif tmp_time[j] != 0:
                if tmp_time[j] > max_pulse_time[j]:
                    max_pulse_time[j] = tmp_time[j]
                if tmp_flow[j] > max_pulse_flow[j]:
                    max_pulse_flow[j] = tmp_flow[j]

                tmp_time[j] = 0
                tmp_flow[j] = 0.0
                pulse_count[j] += 1

    for j in range(len(threshold)):
        if tmp_time[j] > max_pulse_time[j]:
            max_pulse_time[j] = tmp_time[j]
        if tmp_flow[j] > max_pulse_flow[j]:
            max_pulse_flow[j] = tmp_flow[j]

        tmp_time[j] = 0
        tmp_flow[j] = 0.0
        pulse_count[j] += 1 

    return pulse_count, pulse_time, max_pulse_time, pulse_flow, max_pulse_flow  


def draw(flow_data, output_path, mode, SSQ):
    flow_data = np.array(flow_data)
    plt.plot(flow_data[:, 0])
    plt.title('_'.join([mode, 'central', SSQ]))
    plt.savefig(output_path + '_central.png')
    plt.clf()
    plt.plot(flow_data[:, 1])
    plt.title('_'.join([mode, 'peripheral', SSQ]))
    plt.savefig(output_path + '_peripheral.png')
    plt.clf()
    plt.plot(flow_data[:, 0] + flow_data[:, 1])
    plt.title('_'.join([mode, 'total', SSQ]))
    plt.savefig(output_path + '_total.png')
    plt.clf()
    plt.plot(flow_data[:, 2])
    plt.title('_'.join([mode, 'sum', SSQ]))
    plt.savefig(output_path + '_sum.png')
    plt.clf()
    
def demo_video(filename, time):
    cap = cv2.VideoCapture(filename + '.avi')
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')     #works, large
    
    central_out = cv2.VideoWriter(filename + "_central.avi", fourcc, fps, size, True)
    peripheral_out = cv2.VideoWriter(filename + "_peripheral.avi", fourcc, fps, size, True)

    print size
    sep = central_peripheral_seperator(36, (size[1], size[0]))

    i = 1
    while i < (time * fps): 
        ret, frame = cap.read()
        print frame.shape
        central_frame = np.zeros_like(frame)
        peripheral_frame = np.zeros_like(frame)

        central_frame[...,0], peripheral_frame[...,0] = sep.seperate(frame[...,0])
        central_frame[...,1], peripheral_frame[...,1] = sep.seperate(frame[...,1])
        central_frame[...,2], peripheral_frame[...,2] = sep.seperate(frame[...,2])

        central_out.write(central_frame)
        peripheral_out.write(peripheral_frame)

    cap.release()
    central_out.release()
    peripheral_out.release()
    i += 1


if __name__ == '__main__':
    args = get_args()
    from_list(args)
    #demo_video(args.filename,30)
