

# THIS IS A DEMO CODE FOR THE FOLLOWING PAPER 
# Synthetic Views Generation for Absolute Pose Regression and Virtual Reality, BMVC 2018 
# DEVELOPED BY PULAK PURKAIT, EMAIL: pulak.isi@gmail.com

import argparse
import math
# import h5py
import numpy as np
import scipy as sp
import google.protobuf
import tensorflow as tf
import socket
import importlib
import os
import sys
import helper_VisualSFM_7scene
from multiprocessing import Pool

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

#dataset_train = 'dataset_train.txt' 
#dataset_train = 'dataset_new_train2.txt'
#DataSetSize = 1999 

dataset_train = 'dataset_new_train3D.txt'
DataSetSize = 21989

dataset_test = 'dataset_new_test.txt' 

TestDataSetSize = 999 

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_pose', help='Model name: pointnet_pose or pointnet_cls_basic [default: pointnet_pose]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=2000, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=200, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]') 
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
MODEL_PATH = FLAGS.model_path

BLOCK_SIZE1 = 32 # Block SIZE along the ROWS 
BLOCK_SIZE2 = 32 # Block SIZE along the COLUMNS   

MIN_NUM_POINT = 512

NUM_POINT = BLOCK_SIZE1 * BLOCK_SIZE2

IMZ_SZ1 = 320 
IMZ_SZ2 = 240

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = NUM_POINT

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):

		is_training_pl = tf.placeholder(tf.bool, shape=())
		point_set, pose_set = MODEL.placeholder_inputs(BATCH_SIZE, BLOCK_SIZE1, BLOCK_SIZE2) 

		batch = tf.Variable(0)
		bn_decay = get_bn_decay(batch)
		tf.summary.scalar('bn_decay', bn_decay)

		sigx = tf.Variable(0.0)  # Train the coefficients of translation loss
		sigy = tf.Variable(-3.0) # Train the coefficients of rotation  loss 

		pred, pixel_points = MODEL.get_model(point_set, is_training_pl, bn_decay=bn_decay)
		loss = MODEL.get_loss(pred, pose_set, pixel_points, sigx, sigy)
		tf.summary.scalar('loss', loss)

		# Get training operator
		learning_rate = get_learning_rate(batch)

		tf.summary.scalar('learning_rate', learning_rate)
		if OPTIMIZER == 'momentum':
		    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
		elif OPTIMIZER == 'adam':
		    optimizer = tf.train.AdamOptimizer(learning_rate)
		train_op = optimizer.minimize(loss, global_step=batch)

		# Add ops to save and restore all the variables.
		saver = tf.train.Saver()

		# Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl: True})


	# COMMENT ME TO RESTORE *****************************************************************************************************************
	# saver.restore(sess, MODEL_PATH)  

	ops = {'pointclouds_pl': point_set,
	       'labels_pl': pose_set,
	       'is_training_pl': is_training_pl,
	       'pred': pred,
	       'loss': loss,
	       'train_op': train_op,
	       'merged': merged,
	       'step': batch}

	randsz = 200
	arr = np.arange(TestDataSetSize) # np.random.permutation(TestDataSetSize) 
	images_loc, images_des, poses, camerapara = helper_VisualSFM_7scene.get_data(dataset_test, arr[:randsz]) 
	
	
	hist_points_eval = np.zeros((len(images_loc), BLOCK_SIZE1*BLOCK_SIZE2))      
	hist_points_cum_eval = np.zeros((len(images_loc), BLOCK_SIZE1*BLOCK_SIZE2)).astype(int)   	
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
		hist_points_cum_eval[i][1:] = np.cumsum(hist_points_VL[0])[:BLOCK_SIZE1*BLOCK_SIZE2-1]
		hist_points_eval[i] =  np.reshape(hist_points_VL[0], (1, BLOCK_SIZE1*BLOCK_SIZE2)) 

		hist_points_L = hist_points_L.astype(int)
		idxc = np.argsort(all_points[:hist_points_L[0], 1])
		all_points[:hist_points_L[0], :] = all_points[idxc , :]
		#all_points[:hist_points_L[0], 0] = all_points[:hist_points_L[0], 0] - xedges[0]
		all_des[:hist_points_L[0], :] = all_des[idxc , :]


		# print hist_points_VL[1][0], 100*all_points[:hist_points_L[0], 0], xedges[0]
		for k in range(BLOCK_SIZE1-1):
			idxc = hist_points_L[k]+np.argsort(all_points[hist_points_L[k]:hist_points_L[k+1], 1])
			all_points[hist_points_L[k]:hist_points_L[k+1], :] = all_points[idxc, :]
			all_des[hist_points_L[k]:hist_points_L[k+1], :] = all_des[idxc, :]  

			# print all_points[hist_points_L[k]:hist_points_L[k+1], 0], hist_points_VL[1][k]
			#all_points[hist_points_L[k]:hist_points_L[k+1], 0] = all_points[hist_points_L[k]:hist_points_L[k+1], 0] - xedges[k+1]
			#for j in range(BLOCK_SIZE2):
				#all_points[hist_points_cum[i][k*BLOCK_SIZE2+j]:hist_points_cum[i][k*BLOCK_SIZE2+j+1], 1] = all_points[hist_points_cum[i][k*BLOCK_SIZE2+j]:hist_points_cum[i][k*BLOCK_SIZE2+j+1], 1] - yedges[j]
		images_loc[i] = all_points
		images_des[i] = all_des

		
	
	#saver.save([images_train, poses_train ], os.path.join(LOG_DIR, "data.ckpt"))	
	file_size = len(images_loc) 
	images_test = [] 
	poses_test = [] 
	pose_set = np.zeros((len(poses), len(poses[0]))) 
	i = 0
	while i < file_size:
		images_test.append(np.concatenate((images_loc[i], images_des[i]), axis=1)) 
		# images_test.append(np.concatenate((np.divide(images_loc[i], camerapara[i][0]), np.divide(images_des[i].astype(float), 512.0)), axis=1)) 
		poses_test.append(poses[i]) 
		pose_set[i, :] = poses[i]  
		i = i + 1


	rand_size = 200
	

	for epoch in range(MAX_EPOCH):
		log_string('**** EPOCH %03d ****' % (epoch))
		sys.stdout.flush()
		arr = np.random.permutation(DataSetSize)
		images_loc, images_des, poses_train, camerapara = helper_VisualSFM_7scene.get_data(dataset_train, arr[:rand_size]) 		

		hist_points = np.zeros((len(images_loc), BLOCK_SIZE1*BLOCK_SIZE2))      
		hist_points_cum = np.zeros((len(images_loc), BLOCK_SIZE1*BLOCK_SIZE2)).astype(int)   
		print 'Loc: ', len(images_loc), rand_size	

		for i in range(len(images_loc)):
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
			#all_points[:hist_points_L[0], 0] = all_points[:hist_points_L[0], 0] - xedges[0]
			all_des[:hist_points_L[0], :] = all_des[idxc , :]


			# print hist_points_VL[1][0], 100*all_points[:hist_points_L[0], 0], xedges[0]
			for k in range(BLOCK_SIZE1-1):
				idxc = hist_points_L[k]+np.argsort(all_points[hist_points_L[k]:hist_points_L[k+1], 1])
				all_points[hist_points_L[k]:hist_points_L[k+1], :] = all_points[idxc, :]
				all_des[hist_points_L[k]:hist_points_L[k+1], :] = all_des[idxc, :]  

				# print all_points[hist_points_L[k]:hist_points_L[k+1], 0], hist_points_VL[1][k]
				#all_points[hist_points_L[k]:hist_points_L[k+1], 0] = all_points[hist_points_L[k]:hist_points_L[k+1], 0] - xedges[k+1]
				# print 100*all_points[hist_points_L[k]:hist_points_L[k+1], 0]
				#for j in range(BLOCK_SIZE2):
					#all_points[hist_points_cum[i][k*BLOCK_SIZE2+j]:hist_points_cum[i][k*BLOCK_SIZE2+j+1], 1] = all_points[hist_points_cum[i][k*BLOCK_SIZE2+j]:hist_points_cum[i][k*BLOCK_SIZE2+j+1], 1] - yedges[j]
			images_loc[i] = all_points
			images_des[i] = all_des
 
			

		images_train = []
		for i in range(len(images_loc)): 
			images_train.append(np.concatenate((images_loc[i], images_des[i]), axis=1)) 

		for i in range(10): 
			train_one_epoch(sess, ops, images_train, poses_train, train_writer, hist_points, hist_points_cum)

		    # Save the variables to disk.
		if epoch % 5 == 0:
			save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
			log_string("Model saved in file: %s" % save_path)
			eval_one_epoch(sess, ops, images_train[:randsz], poses_train[:randsz], hist_points[:randsz], hist_points_cum[:randsz]) 
			eval_one_epoch(sess, ops, images_test, poses_test, hist_points_eval, hist_points_cum_eval)

def train_one_epoch(sess, ops, images_train, poses_train, train_writer, hist_points, hist_points_cum):
	""" ops: dict mapping from string to tf ops """
	is_training = True
	file_size = len(images_train)
	point_set = np.zeros((file_size, BLOCK_SIZE1, BLOCK_SIZE2, images_train[0].shape[1]))
	pose_set = np.zeros((file_size, len(poses_train[0]))) 
	
	for i in range(file_size):
		all_points = images_train[i]
		
		no_pts, no_ftr = all_points.shape
		idx = np.random.rand(NUM_POINT,)
		idx = np.multiply(idx, hist_points[i])
		idx2 = [ii for ii, e in enumerate(idx) if e <= 10e-10]
		idx = idx.astype(int) 
		# print hist_points_cum[i], idx, no_pts 
		idx = hist_points_cum[i] + idx
		idx3 = [ii for ii, e in enumerate(idx) if e == no_pts]
		idx[idx3] = no_pts - 1 

		#print no_pts, no_ftr, idx.shape
		points = all_points[idx, :]
		points[idx2, :] = 0.0
		points[idx3, :] = 0.0
		
		point_set[i, :, :, :] = np.reshape(points, (BLOCK_SIZE1, BLOCK_SIZE2, images_train[0].shape[1]))
		# print 100*point_set[i, :2, :, :2]
		pose_set[i, :] = poses_train[i]  

	# point_set = np.delete(point_set, [5, 6, 7], axis=3)
	num_batches = file_size / BATCH_SIZE
	
	total_seen = 0
	loss_sum = 0
	
	#sess.run(tf.global_variables_initializer()) 
	for batch_idx in range(num_batches):
		start_idx = batch_idx * BATCH_SIZE
		end_idx = (batch_idx+1) * BATCH_SIZE

		feed_dict = {ops['pointclouds_pl']: point_set[start_idx:end_idx, :, :, :],
		             ops['labels_pl']: pose_set[start_idx:end_idx, :], 
		             ops['is_training_pl']: is_training,}

		summary, step, _, loss_val = sess.run([ops['merged'], ops['step'],
			ops['train_op'], ops['loss']], feed_dict=feed_dict)

		train_writer.add_summary(summary, step)
		# pred_val = np.argmax(pred_val, 1)

		total_seen += BATCH_SIZE
		loss_sum += loss_val
	
	log_string('mean loss: %f' % (loss_sum / float(num_batches+0.0001)))


def eval_one_epoch(sess, ops, images_train, poses_train, hist_points, hist_points_cum):
	error_cnt = 0
	is_training = False
	total_correct = 0
	total_seen = 0
	loss_sum = 0
	
	file_size = len(images_train) 
	num_batches = file_size / BATCH_SIZE 
	print file_size

	point_set = np.zeros((file_size, BLOCK_SIZE1, BLOCK_SIZE2, images_train[0].shape[1]))

	pose_set = np.zeros((file_size, len(poses_train[0]))) 
	pose_predict = np.zeros((file_size, len(poses_train[0]))) 
	
	for i in range(file_size):
		all_points = images_train[i]
		
		no_pts, no_ftr = all_points.shape
		idx = np.random.rand(NUM_POINT,)
		idx = np.multiply(idx, hist_points[i])
		idx2 = [ii for ii, e in enumerate(idx) if e <= 10e-10]
		idx = idx.astype(int) 
		# print hist_points_cum[i], idx, no_pts 
		idx = hist_points_cum[i] + idx
		idx3 = [ii for ii, e in enumerate(idx) if e == no_pts]
		idx[idx3] = no_pts - 1 

		points = all_points[idx, :]
		points[idx2, :] = 0.0
		points[idx3, :] = 0.0

		point_set[i, :, :, :] = np.reshape(points, (BLOCK_SIZE1, BLOCK_SIZE2, images_train[0].shape[1]))
		# print 100*point_set[i, :2, :, :2]
		pose_set[i, :] = poses_train[i]  

	for batch_idx in range(num_batches):
		start_idx = batch_idx * BATCH_SIZE
		end_idx = (batch_idx+1) * BATCH_SIZE
		cur_batch_size = end_idx - start_idx

		feed_dict = {ops['pointclouds_pl']: point_set[start_idx:end_idx, :, :, :],
		             ops['labels_pl']: pose_set[start_idx:end_idx, :], 
		             ops['is_training_pl']: is_training,}

		loss_val, pred_val = sess.run([ops['loss'], ops['pred']],
		                          feed_dict=feed_dict)
		pose_predict[start_idx:end_idx, :] = pred_val 
		# print loss_val, pred_val 

	results = np.zeros((file_size, 2))
	#print pose_set, pose_predict
	for i in range(num_batches * BATCH_SIZE): 
		pose_x = pose_set[i, :3]
		predicted_x = pose_predict[i, :3]
		pose_q = pose_set[i, 3:]
		predicted_q = pose_predict[i, 3:]
		q1 = pose_q / np.linalg.norm(pose_q)
		q2 = predicted_q / np.linalg.norm(predicted_q)
		#print 'Translation: ', pose_x, predicted_x, 'Rotation: ', q1, q2
		d = abs(np.sum(np.multiply(q1,q2)))
		theta = 2 * np.arccos(d) * 180/math.pi
		error_x = np.linalg.norm(pose_x-predicted_x)
		results[i,:] = [error_x, theta]
		#print 'Iteration:  ', i, '  Error XYZ (m):  ', error_x, '  Error Q (degrees):  ', theta, 'norm:= ', np.linalg.norm(q1- q2)  
	
	#print time.time() - t
	print "Median Error: "
	print np.median(results, axis=0)
        # print np.mean(results, axis=0) 

if __name__ == "__main__":
    train()
    LOG_FOUT.close()

