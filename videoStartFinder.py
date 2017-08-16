import os
import cv2 
import argparse
import json
import time 

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--video_dir', type=str, help="directory to video")

	return parser.parse_args()

def find_start(video_path):
	start = False
	stop = False
	try:
		cv2.namedWindow('frame')
		cap = cv2.VideoCapture(video_path)
		frames = []
		i = 0
		while True:
			if i > len(frames):
				print("ERROR: skiping frame")
			elif i == len(frames):
				ret, frame = cap.read()
				frames.append(frame)
			else:
				frame = frames[i]

			cv2.imshow('frame', frame)
			command = cv2.waitKey(33)
			if command != 255:
				print command
			if command == ord('s'):
				stop = True
			elif command == ord('a'):
				stop = False

			if not stop:
				i += 1
			else:
				if command == ord('w'):
					i += 1
				elif command == ord('q'):
					i -= 1
				elif command == ord('e'):
					start = i
					break

			if i < 0:
				i = 0
	except:
		print e
		print("start not change")
	finally:
		cap.release()
		cv2.destroyAllWindows()


	return start


def main(args):
	start = {}
	subjects = [o for o in os.listdir(args.video_dir) if os.path.isdir(os.path.join(args.video_dir,o))]
	for s in subjects:
		videos = [v for v in os.listdir(os.path.join(args.video_dir, s)) if v[-4:] == '.mp4']
		start[s] = {}
		for v in videos:
			video_path = os.path.join(args.video_dir, s, v)
			mode = v.split('_')[0]
				
			if mode == 'Teleport':
				continue
			print(video_path)
			start[s][mode] = find_start(video_path)

	json.dump(start, open(os.path.join(args.video_dir, 'start.json'), 'w'))

if __name__ == '__main__':
	args = get_args()
	main(args)
