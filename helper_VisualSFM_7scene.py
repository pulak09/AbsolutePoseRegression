
# THIS IS A DEMO CODE FOR THE FOLLOWING PAPER 
# Synthetic Views Generation for Absolute Pose Regression and Virtual Reality, BMVC 2018 
# DEVELOPED BY PULAK PURKAIT, EMAIL: pulak.purkait@crl.toshiba.co.uk 

from tqdm import tqdm
import struct
import numpy as np
import os.path
import sys
import random
import math
import string
from multiprocessing import Pool

#import cv2
directory = "/media/ppurkait/Data (B):/SevenScene/stairs/"
# directory = "../Dataset/stairs/"
import numpy.matlib

def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def preprocess(images):
	# images_out = [] #final result
	# pool = Pool()
	# for i in tqdm(range(len(images))):
	filename, file_extension = os.path.splitext(images) 
	with open(filename+'.sift', mode='rb') as f: # b is important -> binary
		nameNversion = f.read(8)
	    
		frmt = (f.read(12)) #the number of features
		num, nLocDim, nDesDim = struct.unpack("3i", frmt)
		
	 	if nDesDim != 128: #should be 128 in this case
			raise RuntimeError, 'Keypoint descriptor length invalid (should be 128).' 
		
		
		strlocs = f.read(num*nLocDim*4)
		locs = struct.unpack(str(num*nLocDim)+"f", strlocs)
		#print(locs)
		strdescriptors = f.read(num*nDesDim)  
		descriptors = struct.unpack(str(num*nDesDim)+"B", strdescriptors)
		
		pos = 0
		data_loc = np.zeros((num, (nLocDim)))
		data_des = np.zeros((num, (nDesDim))).astype('uint8') 

		for point in range(num):
			data_loc[point, :] = locs[point*nLocDim:point*nLocDim+nLocDim] 
			data_des[point, :] = descriptors[point*nDesDim:point*nDesDim+nDesDim] 
		#print(data[0, 0])
	inds = indices(data_des[:, 9], lambda x: x != 45) 
	data_des = data_des[inds, :]
	data_loc = data_loc[inds, :]
	return data_loc, data_des.astype('uint8') 


def get_data(dataset, arr):
	poses = []
	camerapara = []
	images = []
	MX = 650000
	count = 0 
	with open(directory+dataset) as f:
		# print(directory+dataset)
		
		next(f)  # skip the 3 header lines
		next(f)
		next(f)
		for line in f:
			# print(line)
			fname,fl,rd,p0,p1,p2,p3,p4,p5,p6 = line.split()
			p0 = float(p0)
			p1 = float(p1)
			fl = float(fl)
			rd = float(rd)
			p2 = float(p2)
			p3 = float(p3)
			p4 = float(p4)
			p5 = float(p5)
			p6 = float(p6)

			poses.append((p0,p1,p2,p3,p4,p5,p6))
			camerapara.append((fl,rd))
			images.append(directory+fname) 
			count = count + 1
			if count > MX:
				break
	poses_sample = [] 
	camerapara_sample =[]
	images_sample = [] 
	for i in range(arr.shape[0]): 
		poses_sample.append(poses[arr[i]]) 
		camerapara_sample.append(camerapara[arr[i]]) 
		images_sample.append(images[arr[i]]) 
		
	pool = Pool(4) 
	images_loc = []
	images_des = []
	for loc, des in pool.map(preprocess, images_sample, chunksize=1): 
		images_loc.append(loc)
		images_des.append(des)
	pool.close() 
	return images_loc, images_des, poses_sample, camerapara_sample



