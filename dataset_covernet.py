import torch
import utils
import pickle
import numpy as np

from matplotlib import pyplot as plt
from utils import Json_Parser
from torch.utils.data.dataset import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction.helper import PredictHelper
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer


class NuSceneDataset_CoverNet(Dataset):
    def __init__(self, train_mode, config_file_name, layers_list=None, color_list=None):
        super().__init__()
        parser = Json_Parser(config_file_name)
        config = parser.load_parser()

        self.device = torch.device(config['LEARNING']['device'] if torch.cuda.is_available() else 'cpu')
        self.dataroot = config['DATASET']['dataset_path']
        self.nuscenes = NuScenes(config['DATASET']['dataset_str'], dataroot=self.dataroot)
        self.helper = PredictHelper(self.nuscenes)

        self.set = config['DATASET']['set']
        self.train_mode = train_mode
        if self.set == 'train':
            self.train_set = get_prediction_challenge_split("train", dataroot=self.dataroot)
            self.val_set = get_prediction_challenge_split("train_val", dataroot=self.dataroot)
            self.mode = 'train'
        else:
            self.train_set = get_prediction_challenge_split("mini_train", dataroot=self.dataroot)
            self.val_set = get_prediction_challenge_split("mini_val", dataroot=self.dataroot)
            self.mode = 'mini'


        if layers_list is None:
            self.layers_list = config['PREPROCESS']['map_layers_list']
        # if color_list is None:
        #     self.color_list = config['PREPROCESS']['color_list']

        self.resolution = config['PREPROCESS']['resolution']         
        self.meters_ahead = config['PREPROCESS']['meters_ahead']
        self.meters_behind = config['PREPROCESS']['meters_behind']
        self.meters_left = config['PREPROCESS']['meters_left']
        self.meters_right = config['PREPROCESS']['meters_right'] 
        self.patch_box = config['PREPROCESS']['patch_box']
        self.patch_angle = config['PREPROCESS']['patch_angle']
        self.canvas_size = config['PREPROCESS']['canvas_size'] 

        self.past_seconds = config['HISTORY']['past_seconds']
        self.future_seconds = config['HISTORY']['future_seconds']

        self.rasterized = config['PREPROCESS']['rasterized']
        if self.rasterized:
            self.static_layer = StaticLayerRasterizer(helper=self.helper, 
                                                layer_names=self.layers_list, 
                                                # colors=self.color_list,
                                                resolution=self.resolution, 
                                                meters_ahead=self.meters_ahead, 
                                                meters_behind=self.meters_behind,
                                                meters_left=self.meters_left, 
                                                meters_right=self.meters_right)
            self.agent_layer = AgentBoxesWithFadedHistory(helper=self.helper, 
                                                    seconds_of_history=self.past_seconds)
            self.input_repr = InputRepresentation(static_layer=self.static_layer, 
                                            agent=self.agent_layer, 
                                            combinator=Rasterizer())     

        self.show_maps = config['PREPROCESS']['show_maps']
        self.save_maps = config['PREPROCESS']['save_maps']

        self.num_max_agent = config['PREPROCESS']['num_max_agent']
        self.mask = config['PREPROCESS']['mask']

        if self.save_maps:
            if self.train_mode:
                utils.save_map(self.train_set, self.set + 'train', self.input_repr)
            else:
                utils.save_map(self.val_set, self.set + 'val', self.input_repr)
        
  

    def __len__(self):
        if self.train_mode:
            return len(self.train_set)
        else:
            return len(self.val_set)


    def select_agents(self, ego_sample_token, mask):
        sample = self.helper.get_annotations_for_sample(ego_sample_token)

        all_agents_in_sample = []
        for i in range(len(sample)):
            # for check consistency
            assert ego_sample_token in sample[i]['sample_token'], "Something went wrong! Check again"

            if 'vehicle' in sample[i]['category_name']:
                all_agents_in_sample.append(sample[i])

        records_in_patch = []
        for i in range(len(all_agents_in_sample)):
            agent_pose_for_check = all_agents_in_sample[i]['translation']
            bool_in_egoframe = (agent_pose_for_check[0] > mask[0]) and (agent_pose_for_check[0] < mask[2]) and \
                                (agent_pose_for_check[1] < mask[1]) and (agent_pose_for_check[1] > mask[3])     # check if each agent is in the frame of ego (Type : BOOL)
            if bool_in_egoframe:
                data = {}
                data['instance_token'] = all_agents_in_sample[i]['instance_token']
                data['sample_token'] = all_agents_in_sample[i]['sample_token']
                data['translation'] = all_agents_in_sample[i]['translation']
                records_in_patch.append(data)
        
        print("Records_in_patch :  ", records_in_patch)
        
        return records_in_patch



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
                                            seconds=self.past_seconds, in_agent_frame=False, just_xy=False)  
        future = self.helper.get_future_for_agent(instance_token=ego_instance_token, sample_token=ego_sample_token, 
                                            seconds=self.future_seconds, in_agent_frame=False, just_xy=False)
        past_poses = utils.get_pose(past)
        future_poses = utils.get_pose(future)

        # Filter certain records (e.g. intersection)
        records_in_patch = self.select_agents(ego_sample_token, self.mask)
        with open("records_in_patch.pickle","wb") as fw:
            pickle.dump(records_in_patch, fw)
            print("[Records_in_patch] saved.")

        #################################### Image processing ####################################
        img = self.input_repr.make_input_representation(instance_token=ego_instance_token, sample_token=ego_sample_token)

        AssertionError (self.rasterized is False and self.show_maps is True), "Check config again! Img can show only when rasterized flag is True"
        if self.show_maps:
            plt.figure('input_representation')
            plt.imshow(img)
            plt.show()


        return {'img'                  : img,                          # Type : np.array
                'instance_token'       : ego_instance_token,           # Type : str
                'sample_token'         : ego_sample_token,             # Type : str
                'ego_state'            : ego_states,                   # Type : np.array([[vel,accel,yaw_rate]]) --> local(ego's coord)   |   Unit : [m/s, m/s^2, rad/sec]    
                'past_global_ego_pos'  : past_poses,                   # Type : np.array([global_x, global_y, global_yaw])
                'future_global_ego_pos': future_poses,                 # Type : np.array([global_x, global_y, global_yaw])
                'records_in_patch'     : records_in_patch              # Type : list
                }

        # When {vel, accel, yaw_rate} is nan, it will be shown as 0 
        # History List of records.  The rows decrease with time, i.e the last row occurs the farthest in the past.
        # Records_in_patch contains datas in certain area such as intersection.

    


if __name__ == "__main__":
    dataset = NuSceneDataset_CoverNet()
    for i in range(dataset.__len__()):
        dataset.__getitem__(i)
    # dataset.__getitem__(0)
    # train_loader = DataLoader(train_set, batch_size=8, shuffle = True, pin_memory = True, num_workers = 4)
