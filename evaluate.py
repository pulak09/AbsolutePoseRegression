
# THIS IS A DEMO CODE FOR THE FOLLOWING PAPER 
# Synthetic Views Generation for Absolute Pose Regression and Virtual Reality, BMVC 2018 
# DEVELOPED BY PULAK PURKAIT, EMAIL: pulak.purkait@crl.toshiba.co.uk 

import tensorflow as tf
import numpy as np
import scipy as sp
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
import math
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import helper_VisualSFM_7scene
import tf_util

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)

dataset_test = 'dataset_new_test.txt' 
DataSetSize = 999

parser = argparse.ArgumentParser()
#parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_pose', help='Model name: pointnet_pose')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log_trainned/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--visu', type=bool, default=False, help='Whether to dump image for error case [default: False]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
#GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BLOCK_SIZE1 = 32 # Block SIZE along ROWS 
BLOCK_SIZE2 = 32 # Block SIZE along COLUMNS   

NUM_POINT = BLOCK_SIZE1 * BLOCK_SIZE2

IMZ_SZ1 = 320 
IMZ_SZ2 = 240

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate():
	is_training = False

	# with tf.device('/gpu:'+str(GPU_INDEX)):
	point_set, pose_set = MODEL.placeholder_inputs(BATCH_SIZE, BLOCK_SIZE1, BLOCK_SIZE2)
	is_training_pl = tf.placeholder(tf.bool, shape=())

	sigx = tf.Variable(-3.0)
	sigy = tf.Variable(0.0)
	# simple model
	pred, pixel_points = MODEL.get_model(point_set, is_training_pl)
	loss = MODEL.get_loss(pred, pose_set, pixel_points, sigx, sigy)

	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()
        
	# Create a session
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	config.log_device_placement = True
	sess = tf.Session(config=config)

	# Restore variables from disk.
	saver.restore(sess, MODEL_PATH)
	#log_string("Model restored.")

	ops = {'pointclouds_pl': point_set,
	   'labels_pl': pose_set,
	   'is_training_pl': is_training_pl,
	   'pred': pred,
	   'loss': loss}

   	# READ IMAGES 
	arr = np.random.permutation(DataSetSize) 
	images_loc, images_des, poses, camerapara = helper_VisualSFM_7scene.get_data(dataset_test, arr) 
	
	
	hist_points = np.zeros((len(images_loc), BLOCK_SIZE1*BLOCK_SIZE2))      
	hist_points_cum = np.zeros((len(images_loc), BLOCK_SIZE1*BLOCK_SIZE2)).astype(int)   	
	for i in range(len(images_loc)):
		all_points = images_loc[i] 
		all_des = images_des[i] 

		all_points = images_loc[i] 
		all_des = images_des[i] 
		all_points[:, 0] = np.divide(all_points[:, 0]-IMZ_SZ1, camerapara[i][0]) 
		all_points[:, 1] = np.divide(all_points[:, 1]-IMZ_SZ2, camerapara[i][0]) 

		all_points[:, 2] = np.log(1+all_points[:, 3])  
		all_points[:, 3] = np.sin(all_points[:, 4]) 
		all_points[:, 4] = np.cos(all_points[:, 4])   

		# print all_points[0, :6]
		idxr = np.argsort(all_points[:, 0])
		all_points = all_points[idxr, :] 
		all_des = all_des[idxr, :]

		# print  all_points[:10, :2], camerapara[i][0] 
		xedges = np.linspace(-IMZ_SZ1/camerapara[i][0], IMZ_SZ1/camerapara[i][0], num=BLOCK_SIZE1+1) 
		yedges = np.linspace(-IMZ_SZ2/camerapara[i][0], IMZ_SZ2/camerapara[i][0], num=BLOCK_SIZE2+1) 
		# print xedges, yedges
		hist_points_VL = np.histogram2d(all_points[:, 0], all_points[:, 1], bins=(xedges, yedges))
		hist_points_L = np.cumsum(np.sum(hist_points_VL[0], axis=1)).astype(int)  
		hist_points_cum[i][1:] = np.cumsum(hist_points_VL[0])[:BLOCK_SIZE1*BLOCK_SIZE2-1]
		hist_points[i] =  np.reshape(hist_points_VL[0], (1, BLOCK_SIZE1*BLOCK_SIZE2)) 

		hist_points_L = hist_points_L.astype(int)
		idxc = np.argsort(all_points[:hist_points_L[0], 1])
		all_points[:hist_points_L[0], :] = all_points[idxc , :]
		all_points[:hist_points_L[0], 0] = all_points[:hist_points_L[0], 0] - xedges[0]
		all_des[:hist_points_L[0], :] = all_des[idxc , :]


		# print hist_points_VL[1][0], 100*all_points[:hist_points_L[0], 0], xedges[0]
		for k in range(BLOCK_SIZE1-1):
			idxc = hist_points_L[k]+np.argsort(all_points[hist_points_L[k]:hist_points_L[k+1], 1])
			all_points[hist_points_L[k]:hist_points_L[k+1], :] = all_points[idxc, :]
			all_des[hist_points_L[k]:hist_points_L[k+1], :] = all_des[idxc, :]  

			# print all_points[hist_points_L[k]:hist_points_L[k+1], 0], hist_points_VL[1][k]
			all_points[hist_points_L[k]:hist_points_L[k+1], 0] = all_points[hist_points_L[k]:hist_points_L[k+1], 0] - xedges[k+1]
			# print 100*all_points[hist_points_L[k]:hist_points_L[k+1], 0]
			#for j in range(BLOCK_SIZE2):
			#	all_points[hist_points_cum[i][k*BLOCK_SIZE2+j]:hist_points_cum[i][k*BLOCK_SIZE2+j+1], 1] = all_points[hist_points_cum[i][k*BLOCK_SIZE2+j]:hist_points_cum[i][k*BLOCK_SIZE2+j+1], 1] - yedges[j]

		images_loc[i] = all_points
		images_des[i] = all_des
		
	#saver.save([images_train, poses_train ], os.path.join(LOG_DIR, "data.ckpt"))	
	file_size = len(images_loc) 
	images_test = [] 
	poses_test = [] 
	pose_set = np.zeros((len(poses), len(poses[0]))) 
	i = 0
	while i < file_size:
		images_test.append(np.concatenate((images_loc[i],images_des[i].astype(float)), axis=1)) 
		poses_test.append(poses[i]) 
		pose_set[i, :] = poses[i]  
		i = i + 1

	no_evaluation = 1
	pose_predict = np.zeros((no_evaluation, len(images_test), len(poses_test[0]))) 
	for k in range(no_evaluation): 
		pose_predict[k, :, :] = eval_one_epoch(sess, ops, images_test, poses_test, hist_points, hist_points_cum)
	#pose_predict = np.median(pose_predict, axis=0) 

	results = np.zeros((no_evaluation, len(images_test), 2))
	#print pose_set, pose_predict 
	for k in range(no_evaluation): 
		for i in range(len(images_test)):
			pose_x = pose_set[i, :3]
			predicted_x = pose_predict[k, i, :3]
			#pose_q = np.zeros(4)
			#pose_q[3] = 1; 
			pose_q = pose_set[i, 3:]
			#predicted_q = np.zeros(4)
			#predicted_q[3] = 1;
			predicted_q = pose_predict[k, i, 3:]
			q1 = pose_q / np.linalg.norm(pose_q)
			q2 = predicted_q / np.linalg.norm(predicted_q)
			#print 'Translation: ', pose_x, predicted_x, 'Rotation: ', q1, q2
			d = abs(np.sum(np.multiply(q1,q2)))
			theta = 2 * np.arccos(d) * 180/math.pi
			error_x = np.linalg.norm(pose_x-predicted_x)
			results[k, i, :] = [error_x, theta]
		
		#print 'Iteration:  ', i, '  Error XYZ (m):  ', error_x, '  Error Q (degrees):  ', theta, 'norm:= ', np.linalg.norm(q1- q2)  
	results = np.median(results, axis=0) 
	#print time.time() - t
	print np.median(results, axis=0)
        # print np.mean(results, axis=0) 
   
def eval_one_epoch(sess, ops, images_test, poses_test, hist_points, hist_points_cum, topk=1):
	error_cnt = 0
	is_training = False
	total_correct = 0
	total_seen = 0
	loss_sum = 0


	file_size = len(images_test)
	num_batches = file_size / BATCH_SIZE
	# print file_size

	point_set = np.zeros((len(images_test), BLOCK_SIZE1, BLOCK_SIZE2, images_test[0].shape[1])) 
	pose_set = np.zeros((len(images_test), len(poses_test[0]))) 
	pose_predict = np.zeros((len(images_test), len(poses_test[0]))) 
	
	for i in range(file_size):
		all_points = images_test[i]
		
		no_pts, no_ftr = all_points.shape
		idx = np.random.rand(NUM_POINT,)
		idx = np.multiply(idx, hist_points[i])
		idx2 = [ii for ii, e in enumerate(idx) if e <= 10e-10]
		idx = idx.astype(int) 
		#print hist_points_cum[i], idx, no_pts 
		idx = hist_points_cum[i] + idx
		idx3 = [ii for ii, e in enumerate(idx) if e == no_pts]
		idx[idx3] = no_pts - 1 

		points = all_points[idx, :]
		points[idx2, :] = 0.0; 
		points[idx3, :] = 0.0; 

		point_set[i, :, :, :] = np.reshape(points, (BLOCK_SIZE1, BLOCK_SIZE2, images_test[0].shape[1]))
		#print point_set[i, :, :, :2]
		pose_set[i, :] = poses_test[i]  


	for batch_idx in range(num_batches):
		start_idx = batch_idx * BATCH_SIZE
		end_idx = (batch_idx+1) * BATCH_SIZE
		cur_batch_size = end_idx - start_idx

		feed_dict = {ops['pointclouds_pl']: point_set[start_idx:end_idx, :, :],
		             ops['labels_pl']: pose_set[start_idx:end_idx, :], 
		             ops['is_training_pl']: is_training,}

		loss_val, pred_val = sess.run([ops['loss'], ops['pred']],
		                          feed_dict=feed_dict)
		pose_predict[start_idx:end_idx, :] = pred_val 
                print(batch_idx) 
		# print loss_val, pred_val 

	return pose_predict

if __name__=='__main__':
    with tf.Graph().as_default():
	evaluate()
    LOG_FOUT.close()


