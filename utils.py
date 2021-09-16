# Code written by govvijaycal from Github
import os
import numpy as np
from PIL import Image
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion.quaternion import Quaternion

def data_filter(data):
	for i in range(len(data)):
		nan_check = np.isnan(data[i])
		if nan_check:
			data[i] = 0
			
	return data

def save_map(self, dataset, type, input_repr):

	print("starting to save maps")

	for i, _ in enumerate(dataset): 
		instance_token_img, sample_token_img = dataset[i].split('_')
		
		folder_path = os.path.join(self.dataroot, 'saved_map', type)
		if not os.path.exists(folder_path):
			os.makedirs(folder_path, exist_ok=True)

		file_path = os.path.join(folder_path,"maps_{0}.jpg".format(i))

		instance_token_img, sample_token_img = dataset[i].split('_')
		img = input_repr.make_input_representation(instance_token_img, sample_token_img)
		im = Image.fromarray(img)
		im.save(file_path)
	
		print("Map saving process : [{0}/{1}] completed".format(i, len(dataset)),end='\r')
	
	print("done saving maps")


def get_pose_from_annot(annotation):
    x, y, _ = annotation['translation']
    yaw = quaternion_yaw(Quaternion(annotation['rotation']))
    
    return [x, y, yaw]

def get_pose(annotation_list):
    return np.array([get_pose_from_annot(ann) for ann in annotation_list])


def rotation_global_to_local(yaw):
    return np.array([[ np.cos(yaw), np.sin(yaw)], \
                [-np.sin(yaw), np.cos(yaw)]])


def angle_mod_2pi(angle):
	return (angle + np.pi) % (2.0 * np.pi) - np.pi


def pose_diff_norm(pose_diff):
	# Not exactly a traditional norm but just meant to ensure no pose differences.
	xy_norm    = np.linalg.norm(pose_diff[:,:2], ord=np.inf)
	angle_norm = np.max( [angle_mod_2pi(x) for x in pose_diff[:,2]] )
	return xy_norm + angle_norm



def convert_global_to_local_forhistory(global_pose_origin, global_poses):
	R_global_to_local = rotation_global_to_local(global_pose_origin[2])
	t_global_to_local = - R_global_to_local @ global_pose_origin[:2]

	local_xy  = np.array([ R_global_to_local @ pose[:2] + t_global_to_local 
	                         for pose in global_poses])

	local_yaw = np.array([ angle_mod_2pi(pose[2] - global_pose_origin[2])
		                     for pose in global_poses])
	
	return np.column_stack((local_xy, local_yaw))


def convert_global_to_local_forpose(global_pose_origin, global_pose):
    R_global_to_local = rotation_global_to_local(global_pose_origin[2])
    t_global_to_local = - R_global_to_local @ global_pose_origin[:2]

    local_xy  = R_global_to_local @ global_pose[:2] + t_global_to_local 
    local_yaw = angle_mod_2pi(global_pose[2] - global_pose_origin[2])
    output = [local_xy[0], local_xy[1], local_yaw]

    return output
