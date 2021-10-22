import os
import json
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.prediction.input_representation.interface import Combinator
from pyquaternion.quaternion import Quaternion


class Json_Parser:
    def __init__(self, file_name):
        with open(file_name) as json_file:
            self.config = json.load(json_file)

    def load_parser(self):
        return self.config


def data_filter(data):
	for i in range(len(data)):
		nan_check = np.isnan(data[i])
		if nan_check:
			data[i] = 0
			
	return data

def save_imgs(self, dataset, type, input_repr):
	print("starting to save maps")

	for i, _ in enumerate(dataset): 
		instance_token_img, sample_token_img = dataset[i].split('_')
		
		folder_path = os.path.join(self.dataroot, 'saved_img', type)
		if not os.path.exists(folder_path):
			os.makedirs(folder_path, exist_ok=True)

		file_path = os.path.join(folder_path,"img_{0}.jpg".format(i))

		instance_token_img, sample_token_img = dataset[i].split('_')
		img = input_repr.make_input_representation(instance_token_img, sample_token_img)
		im = Image.fromarray(img)
		im.save(file_path)
	
		print("Img saving process : [{0}/{1}] completed".format(i, len(dataset)),end='\r')
	print("done saving imgs")


def save_maps(self, type, map, idx):
	folder_path = os.path.join(self.dataroot, 'saved_map', type)
	if not os.path.exists(folder_path):
		os.makedirs(folder_path, exist_ok=True)
	file_path = os.path.join(folder_path,"maps_{0}.jpg".format(idx))

	plt.savefig(file_path)	
	print("done saving map_{}".format(idx))


def get_pose_from_annot(annotation):
    x, y, _ = annotation['translation']
    yaw = quaternion_yaw(Quaternion(annotation['rotation']))
    
    return [x, y, yaw]

def get_pose(annotation_list):
    return np.array([get_pose_from_annot(ann) for ann in annotation_list])


def get_pose2(annotation_list, max_size):
	temp = []
	num = max_size - len(annotation_list)

	for idx in range(len(annotation_list)):
		temp.append(get_pose_from_annot(annotation_list[idx]))

	for i in range(num):
		temp.append([0,0,0])

	return np.array(temp)


def check_shape(check_list, max_size, dim):
	# input type is np.array

	num = len(check_list) - max_size
	check_shape = np.shape(check_list)

	if num != 0:
		for i in range(abs(num)):
			if dim == 1:
				check_list = np.append(check_list, np.array([0]))
			elif dim == 2:
				empty = np.zeros([1, 3])
				check_list = np.append(check_list, empty, axis=0)
			elif dim == 3:
				empty = np.zeros([1, check_shape[1], check_shape[2]])
				check_list = np.append(check_list, empty, axis=0)
	
	return check_list
	

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

if __name__ == "__main__":
	x = [[14.390376776648509, -10.52232451628899, 1.8050195123977728], [-11.560309700592597, -0.24055067322350965, 0.016563174589137475], [-32.956026099823475, 8.602587500732682, 0.10983356983239645], [-1.702659006289423, 3.1352247620069846, 0.017453292519943986], [0.0, 0.0, 0.0], [-7.504870698154491, 8.325104672253829, 0.09510299092888275]]
	print(np.shape(np.array(x)))
	# output = check_shape(x,10,True)
	em = np.array([[0,0,0]])
	print(np.shape(em))
	output = np.append(x, em, axis=0)
	print(output)
	print(np.shape(output))