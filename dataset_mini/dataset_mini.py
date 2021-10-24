import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import utils
import torch
import intersection_dataload
import numpy as np

from utils import Json_Parser
from matplotlib import pyplot as plt
from torch.utils.data.dataset import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction.helper import PredictHelper
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer


class NuSceneDataset_Mini(Dataset):
    def __init__(self, train_mode, config_file_name, layers_list=None, color_list=None):
        super().__init__()
        parser = Json_Parser(config_file_name)
        config = parser.load_parser()

        self.device = torch.device(config['LEARNING']['device'] if torch.cuda.is_available() else 'cpu')
        self.dataroot = config['DATASET']['dataset_path']

        self.intersection_use= config['DATASET']['intersection_use']        # only available for mini_dataset
        self.img_preprocess = config['DATASET']['img_preprocess']

        self.nuscenes = NuScenes(config['DATASET']['dataset_str'], dataroot=self.dataroot)
        self.helper = PredictHelper(self.nuscenes)

        self.set = config['DATASET']['set']
        self.train_mode = train_mode
        if self.set == 'train':
            self.mode = 'train'
            if self.train_mode:
                self.train_set = get_prediction_challenge_split("train", dataroot=self.dataroot)
            else:
                self.val_set = get_prediction_challenge_split("train_val", dataroot=self.dataroot)
        else:            
            self.mode = 'mini'
            if self.train_mode:
                self.train_set = get_prediction_challenge_split("mini_train", dataroot=self.dataroot)
            else:
                self.val_set = get_prediction_challenge_split("mini_val", dataroot=self.dataroot)
            if self.intersection_use:
                if self.train_mode:
                    self.train_set = intersection_dataload.token_save(self.train_set)
                else:
                    self.val_set = intersection_dataload.token_save(self.val_set)
                
                
        if layers_list is None:
            self.layers_list = config['PREPROCESS']['img_layers_list']
        if color_list is None:
            self.color_list = []
            for i in range(len(self.layers_list)):
                self.color_list.append((255,255,255))

        self.resolution = config['PREPROCESS']['resolution']         
        self.meters_ahead = config['PREPROCESS']['meters_ahead']
        self.meters_behind = config['PREPROCESS']['meters_behind']
        self.meters_left = config['PREPROCESS']['meters_left']
        self.meters_right = config['PREPROCESS']['meters_right'] 

        self.num_past_hist = int(config['HISTORY']['num_past_hist']/2)
        self.num_future_hist = int(config['HISTORY']['num_future_hist']/2)

        self.static_layer = StaticLayerRasterizer(helper=self.helper, 
                                            layer_names=self.layers_list, 
                                            colors=self.color_list,
                                            resolution=self.resolution, 
                                            meters_ahead=self.meters_ahead, 
                                            meters_behind=self.meters_behind,
                                            meters_left=self.meters_left, 
                                            meters_right=self.meters_right)
        self.agent_layer = AgentBoxesWithFadedHistory(helper=self.helper, 
                                                seconds_of_history=self.num_past_hist)
        self.input_repr = InputRepresentation(static_layer=self.static_layer, 
                                        agent=self.agent_layer, 
                                        combinator=Rasterizer())     

        self.show_imgs = config['PREPROCESS']['show_imgs']
        self.save_imgs = config['PREPROCESS']['save_imgs']

        self.num_max_agent = config['PREPROCESS']['num_max_agent']

        if (self.mode == 'mini') and self.img_preprocess:
            if self.train_mode:
                utils.save_imgs(self, self.train_set, self.set + 'train', self.input_repr)
            else:
                utils.save_imgs(self, self.val_set, self.set + 'val', self.input_repr)
        
        if self.save_imgs:
            if self.train_mode:
                utils.save_imgs(self, self.train_set, self.set + 'train', self.input_repr)
            else:
                utils.save_imgs(self, self.val_set, self.set + 'val', self.input_repr)
        
  
    def __len__(self):
        if self.train_mode:
            return len(self.train_set)
        else:
            return len(self.val_set)


    def __getitem__(self, idx):
        if self.train_mode:
            self.dataset = self.train_set
        else:
            self.dataset = self.val_set

        #################################### State processing ####################################
        ego_instance_token, ego_sample_token = self.dataset[idx].split('_')

        ego_vel = self.helper.get_velocity_for_agent(ego_instance_token, ego_sample_token)
        ego_accel = self.helper.get_acceleration_for_agent(ego_instance_token, ego_sample_token)
        ego_yawrate = self.helper.get_heading_change_rate_for_agent(ego_instance_token, ego_sample_token)

        # Filter unresonable data (make nan to zero)
        [ego_vel, ego_accel, ego_yawrate] = utils.data_filter([ego_vel, ego_accel, ego_yawrate])        
        ego_states = np.array([[ego_vel, ego_accel, ego_yawrate]])

        # GLOBAL history
        past = self.helper.get_past_for_agent(instance_token=ego_instance_token, sample_token=ego_sample_token, 
                                            seconds=self.num_past_hist, in_agent_frame=False, just_xy=False)  
        future = self.helper.get_future_for_agent(instance_token=ego_instance_token, sample_token=ego_sample_token, 
                                            seconds=self.num_future_hist, in_agent_frame=False, just_xy=False)
        future_poses_m = utils.get_pose2(future, self.num_future_hist)
        num_future_mask = len(future)


        #################################### Image processing ####################################
        img = self.input_repr.make_input_representation(instance_token=ego_instance_token, sample_token=ego_sample_token)
        if self.show_imgs:
            plt.figure('input_representation')
            plt.imshow(img)
            plt.show()


        return {'img'                  : img,                          # Type : np.array
                'ego_state'            : ego_states,                   # Type : np.array([[vel,accel,yaw_rate]]) --> local(ego's coord)   |   Unit : [m/s, m/s^2, rad/sec]    
                'num_future_mask'      : num_future_mask,              # Type : int .. indicate the number of future history for masking 'future_global_ego_pos'
                'future_global_ego_pos': future_poses_m,               # Type : np.array([global_x, global_y, global_yaw])
                }

        # When {vel, accel, yaw_rate} is nan, it will be shown as 0 
        # History List of records.  The rows decrease with time, i.e the last row occurs the farthest in the past.

