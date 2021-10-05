import torch
import utils
import numpy as np

from matplotlib import pyplot as plt
from config import Config
from torch.utils.data.dataset import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction.helper import PredictHelper
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory

from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer




class NuSceneDataset(Dataset):
    def __init__(self, train_mode):
        super().__init__()
        config = Config()

        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.dataroot = config.dataset_path
        self.nuscenes = NuScenes(config.dataset_str, dataroot=self.dataroot)
        self.helper = PredictHelper(self.nuscenes)

        self.set = config.set
        self.train_mode = train_mode
        if self.set == 'train':
            self.train_set = get_prediction_challenge_split("train", dataroot=self.dataroot)
            self.val_set = get_prediction_challenge_split("train_val", dataroot=self.dataroot)
            self.mode = 'train'
        else:
            self.train_set = get_prediction_challenge_split("mini_train", dataroot=self.dataroot)
            self.val_set = get_prediction_challenge_split("mini_val", dataroot=self.dataroot)
            self.mode = 'mini'

        self.layers_list = config.map_layers_list
        self.color_list = config.color_list

        self.resolution = config.resolution                 
        self.meters_ahead = config.meters_ahead
        self.meters_behind = config.meters_behind
        self.meters_left = config.meters_left 
        self.meters_right = config.meters_right 
        self.patch_angle = config.patch_angle

        self.past_seconds = config.past_seconds 
        self.future_seconds = config.future_seconds 

        self.rasterized = config.rasterized
        if self.rasterized:
            self.static_layer = StaticLayerRasterizer(helper=self.helper, 
                                                layer_names=self.layers_list, 
                                                colors=self.color_list,
                                                resolution=self.resolution, 
                                                meters_ahead=self.meters_ahead, 
                                                meters_behind=self.meters_behind,
                                                meters_left=self.meters_left, 
                                                meters_right=self.meters_right)
            self.agent_layer = AgentBoxesWithFadedHistory(helper=self.helper, 
                                                seconds_of_history=self.past_seconds, 
                                                resolution=self.resolution, 
                                                meters_ahead=self.meters_ahead, 
                                                meters_behind=self.meters_behind,
                                                meters_left=self.meters_left, 
                                                meters_right=self.meters_right)
            self.input_repr = InputRepresentation(static_layer=self.static_layer, 
                                                agent=self.agent_layer, 
                                                combinator=Rasterizer())     

        self.show_maps = config.show_maps
        self.save_maps = config.save_maps

        self.num_max_agent = config.num_max_agent
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


    def select_agents(self, ego_sample_token, ego_pose):
        sample = self.helper.get_annotations_for_sample(ego_sample_token)
        mask = [ego_pose[0] - self.meters_left, ego_pose[1] + self.meters_ahead, ego_pose[0] + self.meters_right, ego_pose[1] - self.meters_behind] # [top_left_x, top_left_y, bottom_right_x, bottom_right_y]

        all_agents_in_sample = []
        for i in range(len(sample)):
            # for check consistency
            assert ego_sample_token in sample[i]['sample_token'], "Something went wrong! Check again"

            if 'vehicle' in sample[i]['category_name']:
                all_agents_in_sample.append(sample[i])

        # 1) position filtering
        agents_in_egoframe = []
        dist_to_ego = []
        for i in range(len(all_agents_in_sample)):
            agent_pose_for_check = all_agents_in_sample[i]['translation']
            bool_in_egoframe = (agent_pose_for_check[0] > mask[0]) and (agent_pose_for_check[0] < mask[2]) and \
                                (agent_pose_for_check[1] < mask[1]) and (agent_pose_for_check[1] > mask[3])     # check if each agent is in the frame of ego (Type : BOOL)
            if bool_in_egoframe:
                agents_in_egoframe.append(all_agents_in_sample[i])
                dist = np.linalg.norm(np.array((ego_pose[0],ego_pose[1])) - np.array((agent_pose_for_check[0],agent_pose_for_check[1])))
                dist_to_ego.append(dist)

        # 2) num_agents filtering (Up to num_max_agent in config.py)
        num_agents = len(agents_in_egoframe)        # before filtering
        agents = []
        if num_agents > self.num_max_agent:
            sort_idx_list = sorted(range(len(agents_in_egoframe)), key=lambda k: agents_in_egoframe[k])
            put_list = sort_idx_list[:self.num_max_agent]
            for i in range(len(agents_in_egoframe)):
                if i in put_list:
                    agents.append(agents_in_egoframe[i])
        else:
            agents = agents_in_egoframe
        num_agents = len(agents)                    # after filtering

        return num_agents, agents


    def __getitem__(self, idx):
        """
        Select ego vehicle using idx
        Represent states of agents nearby ego vehicle
        """
        if self.train_mode:
            self.dataset = self.train_set
        else:
            self.dataset = self.val_set

        #################################### Ego states ####################################
        ego_instance_token, ego_sample_token = self.dataset[idx].split("_")
        print(idx, "th : ")
        print(ego_instance_token, ego_sample_token)
        ego_annotation = self.helper.get_sample_annotation(ego_instance_token, ego_sample_token)

        ego_pose = np.array(utils.get_pose_from_annot(ego_annotation))
        ego_vel = self.helper.get_velocity_for_agent(ego_instance_token, ego_sample_token)
        ego_accel = self.helper.get_acceleration_for_agent(ego_instance_token, ego_sample_token)
        ego_yawrate = self.helper.get_heading_change_rate_for_agent(ego_instance_token, ego_sample_token)

        # Filter unresonable data (make nan to zero)
        [ego_vel, ego_accel, ego_yawrate] = utils.data_filter([ego_vel, ego_accel, ego_yawrate])        
        ego_states = np.array([ego_vel, ego_accel, ego_yawrate])


        # GLOBAL history
        past = self.helper.get_past_for_agent(instance_token=ego_instance_token, sample_token=ego_sample_token, 
                                            seconds=self.past_seconds, in_agent_frame=False, just_xy=False)  
        future = self.helper.get_future_for_agent(instance_token=ego_instance_token, sample_token=ego_sample_token, 
                                            seconds=self.future_seconds, in_agent_frame=False, just_xy=False)
        past_poses = utils.get_pose(past)
        future_poses = utils.get_pose(future)

        # Get sampling time between current and past/future sample (since it is not constant sampling)
        current_time = self.helper._timestamp_for_sample(ego_sample_token)                                  # Unit : microsec
        past_time = [self.helper._timestamp_for_sample(p['sample_token']) for p in past]
        future_time = [self.helper._timestamp_for_sample(f['sample_token']) for f in future]

        past_cur_time_diff = np.array([pow(10,-6) * (p - current_time) for p in past_time])
        future_cur_time_diff = np.array([pow(10,-6) * (f - current_time) for f in future_time])

        #################################### Agent states ####################################
        # Select agents nearby Ego
        num_agents, agents_list = self.select_agents(ego_sample_token=ego_sample_token, ego_pose=ego_pose)

        agent_local_pose_list = []
        agent_states_list = []
        agent_past_local_poses_list = []
        agent_future_local_poses_list = []
        num_agent_past_hist = []
        num_agent_future_hist = []

        for i in range(num_agents):
            assert num_agents == len(agents_list), "num_agents != len(agents_list)"
            assert ego_sample_token == agents_list[i]['sample_token'], "agents sample token != ego sample token"

            instance_token_agent = agents_list[i]['instance_token']
            sample_token_agent = agents_list[i]['sample_token']
            agent_annotation = self.helper.get_sample_annotation(instance_token_agent, sample_token_agent)

            agent_pose = utils.get_pose_from_annot(agent_annotation)
            agent_local_pose  = np.array(utils.convert_global_to_local_forpose(ego_pose, agent_pose))
            agent_local_pose_list.append(agent_local_pose)
            agent_vel = self.helper.get_velocity_for_agent(instance_token_agent, sample_token_agent)
            agent_accel = self.helper.get_acceleration_for_agent(instance_token_agent, sample_token_agent)
            agent_yawrate = self.helper.get_heading_change_rate_for_agent(ego_instance_token, ego_sample_token)
            
            # Filter unresonable data (make nan to zero)
            [agent_vel, agent_accel, agent_yawrate] = utils.data_filter([agent_vel, agent_accel, agent_yawrate])        
            agent_states = np.array([agent_vel, agent_accel, agent_yawrate])
            agent_states_list.append(agent_states)

            # Agent LOCAL history 
            past_global_poses = self.helper.get_past_for_agent(instance_token=instance_token_agent, sample_token=sample_token_agent, 
                                                seconds=self.past_seconds, in_agent_frame=False, just_xy=False)  

            future_global_poses = self.helper.get_future_for_agent(instance_token=instance_token_agent, sample_token=sample_token_agent, 
                                                seconds=self.future_seconds, in_agent_frame=False, just_xy=False) 

            past_global_poses = utils.get_pose(past_global_poses)
            future_global_poses = utils.get_pose(future_global_poses)

            agent_past_local_poses = utils.convert_global_to_local_forhistory(ego_pose, past_global_poses)
            agent_past_local_poses_list.append(agent_past_local_poses)
            num_agent_past_hist.append(len(agent_past_local_poses))
            agent_future_local_poses = utils.convert_global_to_local_forhistory(ego_pose, future_global_poses)
            agent_future_local_poses_list.append(agent_future_local_poses)
            num_agent_future_hist.append(len(agent_future_local_poses))

        #################################### Image processing ####################################
        img = self.input_repr.make_input_representation(instance_token=ego_instance_token, sample_token=ego_sample_token)

        AssertionError (self.rasterized is False and self.show_maps is True), "Check config again! Img can show only when rasterized flag is True"
        if self.show_maps:
            plt.figure('input_representation_{}'.format(idx))
            plt.imshow(img)
            plt.show()


        return {'img'                  : img,                          # Type : np.array
                # ego vehicle                   
                'instance_token'       : ego_instance_token,           # Type : str
                'sample_token'         : ego_sample_token,             # Type : str
                'ego_cur_pos'          : ego_pose,                     # Type : list [x,y,yaw] --> global 
                'ego_state'            : ego_states,                   # Type : np.array([vel,accel,yaw_rate]) --> local(ego's coord)   | Unit : [m/s, m/s^2, rad/sec]    
                'past_global_ego_pos'  : past_poses,                   # Type : np.array([global_x, global_y, global_yaw])
                'future_global_ego_pos': future_poses,                 # Type : np.array([global_x, global_y, global_yaw])
                'past_cur_diff_ego'    : past_cur_time_diff,           # Difference between current and past sampling time
                'future_cur_diff_ego'  : future_cur_time_diff,         # Difference between current and future sampling time
                # agents nearby ego (all local data in ego coordinate)
                'num_agents'           : num_agents,
                'agent_cur_pose'       : np.array(agent_local_pose_list),                           # Type : np.row_stack([local_x, local_y, local_yaw])
                'agent_state'          : np.array(agent_states_list),                               # Type : np.row_stack([vel,accel,yaw_rate]) --> local(agent's coord)     | Unit : [m/s, m/s^2, rad/sec]    
                'agent_past_pose'      : np.array(agent_past_local_poses_list, dtype=object),       # Type : np.row_stack([local_x, local_y, local_yaw])                
                'agent_future_pose'    : np.array(agent_future_local_poses_list, dtype=object),     # Type : np.row_stack([local_x, local_y, local_yaw])
                'num_agent_past_hist'  : np.array(num_agent_past_hist),                             # Type : np.array([len_past_history for each agent])  
                'num_agent_future_hist': np.array(num_agent_future_hist)                            # Type : np.array([len_future_history for each agent])
                }

        # When {vel, accel, yaw_rate} is nan, it will be shown as 0 
        # History List of records.  The rows decrease with time, i.e the last row occurs the farthest in the past.

    


if __name__ == "__main__":
    dataset = NuSceneDataset(train_mode=True)
    for i in range(dataset.__len__()):
        dataset.__getitem__(i)

    # train_loader = DataLoader(train_set, batch_size=8, shuffle = True, pin_memory = True, num_workers = 4)
