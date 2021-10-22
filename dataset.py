import torch
import utils
import numpy as np

from matplotlib import pyplot as plt
from config import Config
from torch.utils.data.dataset import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
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

        self.bbox_size = config.bbox_size_limit
        self.img_layers_list = config.img_map_layers_list
        self.map_layers_list = config.map_layers_list
        self.lane_type = config.lane_type
        self.color_list = []
        for i in range(len(self.img_layers_list)):
            self.color_list.append((255,255,255))

        self.canvas_size = config.canvas_size
        self.fig_size = config.fig_size
        self.resolution = config.resolution                 
        self.meters_ahead = config.meters_ahead
        self.meters_behind = config.meters_behind
        self.meters_left = config.meters_left 
        self.meters_right = config.meters_right 
        self.patch_angle = config.patch_angle

        self.num_past_hist = config.num_past_hist
        self.num_future_hist = config.num_future_hist

        self.static_layer = StaticLayerRasterizer(helper=self.helper, 
                                            layer_names=self.img_layers_list, 
                                            colors=self.color_list,
                                            resolution=self.resolution, 
                                            meters_ahead=self.meters_ahead, 
                                            meters_behind=self.meters_behind,
                                            meters_left=self.meters_left, 
                                            meters_right=self.meters_right)
        self.agent_layer = AgentBoxesWithFadedHistory(helper=self.helper, 
                                            seconds_of_history=self.num_past_hist, 
                                            resolution=self.resolution, 
                                            meters_ahead=self.meters_ahead, 
                                            meters_behind=self.meters_behind,
                                            meters_left=self.meters_left, 
                                            meters_right=self.meters_right)
        self.input_repr = InputRepresentation(static_layer=self.static_layer, 
                                            agent=self.agent_layer, 
                                            combinator=Rasterizer())     
        self.show_imgs = config.show_imgs
        self.save_imgs = config.save_imgs
        self.show_maps = config.show_maps
        self.save_maps = config.save_maps

        self.num_max_agent = config.num_max_agent
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
            sort_idx_list = sorted(range(len(dist_to_ego)), key=lambda k: dist_to_ego[k])
            put_list = sort_idx_list[:self.num_max_agent]
            for i in range(len(agents_in_egoframe)):
                if i in put_list:
                    agents.append(agents_in_egoframe[i])
        else:
            agents = agents_in_egoframe
        num_agents = len(agents)                    # (after) filtering

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
                                            seconds=int(self.num_past_hist/2), in_agent_frame=False, just_xy=False)  
        future = self.helper.get_future_for_agent(instance_token=ego_instance_token, sample_token=ego_sample_token, 
                                            seconds=int(self.num_future_hist/2), in_agent_frame=False, just_xy=False)
        ego_hist_num_mask = [len(past), len(future)]

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
            agent_local_pose  = utils.convert_global_to_local_forpose(ego_pose, agent_pose)
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
                                                seconds=int(self.num_past_hist/2), in_agent_frame=False, just_xy=False)  

            future_global_poses = self.helper.get_future_for_agent(instance_token=instance_token_agent, sample_token=sample_token_agent, 
                                                seconds=int(self.num_future_hist/2), in_agent_frame=False, just_xy=False) 
            
            print("idx : ", i)
            if past_global_poses == [] or future_global_poses == []:
                print("out")
                continue 

            print("num_agent :", num_agents)
            print("11:", past_global_poses)
            past_global_poses = utils.get_pose(past_global_poses)
            future_global_poses = utils.get_pose(future_global_poses)
            print("22:", past_global_poses)

            agent_past_local_poses = utils.convert_global_to_local_forhistory(ego_pose, past_global_poses)            
            agent_future_local_poses = utils.convert_global_to_local_forhistory(ego_pose, future_global_poses)

            num_agent_past_hist.append(len(agent_past_local_poses))
            num_agent_future_hist.append(len(agent_future_local_poses))

            print("before : ",agent_past_local_poses)
            agent_past_local_poses = utils.check_shape(agent_past_local_poses, self.num_past_hist, dim=2)
            agent_future_local_poses = utils.check_shape(agent_future_local_poses, self.num_future_hist, dim=2)
            print("after : ", agent_past_local_poses)
            if i==0:
                agent_past_local_poses_list = np.array(agent_past_local_poses)                
                agent_future_local_poses_list = np.array(agent_future_local_poses)

            else:
                agent_past_local_poses_list = np.concatenate([agent_past_local_poses_list, agent_past_local_poses], axis=0)
                print("shape1: ", np.shape(agent_past_local_poses_list))
                print("shape2: ", np.shape(agent_past_local_poses))          
                agent_future_local_poses_list = np.concatenate([agent_future_local_poses_list, agent_future_local_poses], axis=0)          
                print("shape3: ", np.shape(agent_future_local_poses_list))
                print("shape4: ", np.shape(agent_future_local_poses)) 
            agent_past_local_poses_np = agent_past_local_poses_list.reshape(-1, self.num_past_hist, 3)
            agent_future_local_poses_np = agent_future_local_poses_list.reshape(-1, self.num_future_hist, 3)


        # unify shape of variant data
        past_poses_m = utils.get_pose2(past, self.num_past_hist)
        future_poses_m = utils.get_pose2(future, self.num_future_hist)
        past_cur_time_diff_m = utils.check_shape(past_cur_time_diff, self.num_past_hist, dim=1)
        future_cur_time_diff_m = utils.check_shape(future_cur_time_diff, self.num_future_hist, dim=1)

        agent_local_pose_list_m = utils.check_shape(agent_local_pose_list, self.num_max_agent, dim=2)
        agent_states_list_m = utils.check_shape(agent_states_list, self.num_max_agent, dim=2)
        # agent_past_local_poses_list_m = utils.check_shape(agent_past_local_poses_np, self.num_max_agent, dim=3)        
        # agent_future_local_poses_list_m = utils.check_shape(agent_future_local_poses_np, self.num_max_agent, dim=3)
        num_agent_past_hist_m = utils.check_shape(num_agent_past_hist, self.num_max_agent, dim=1)
        num_agent_future_hist_m = utils.check_shape(num_agent_future_hist, self.num_max_agent, dim=1)

        #################################### Image processing ####################################
        img = self.input_repr.make_input_representation(instance_token=ego_instance_token, sample_token=ego_sample_token)
        if self.show_imgs:
            plt.figure('input_representation_{}'.format(idx))
            plt.imshow(img)
            plt.show()


        # #################################### Map processing ######################################
        # map_location_name = self.helper.get_map_name_from_sample_token(ego_sample_token)
        # nusc_map = NuScenesMap(map_name=map_location_name, dataroot=self.dataroot)
        # closest_lane = nusc_map.get_closest_lane(ego_pose[0], ego_pose[1], radius=2)
        
        # assert (self.lane_type not in ['incoming, outgoing']), "Check lane_type config again! It should be 'incoming' or 'outgoint' but your type is {0}".format(self.lane_type)
        # first_lane = nusc_map._get_connected_lanes(closest_lane, self.lane_type)
        
        # def plot_segment(lane_token):
        #     global from_ego
        #     global start_end_position

        #     outgoing_lane = nusc_map._get_connected_lanes(lane_token, 'outgoing')

        #     if (lane_token==first_lane[0]):
        #         from_ego = 0
        #         start_end_position = np.array([])
        #     else:
        #         from_ego += 1

        #     for i in range(len(outgoing_lane)):
        #         outgoing_lane_info = nusc_map.get_arcline_path(outgoing_lane[i])
        #         x_(after) = outgoing_lane_info[0]['end_pose'][0]
        #         y_(after) = outgoing_lane_info[0]['end_pose'][1]
        #         x_before = outgoing_lane_info[0]['start_pose'][0]
        #         y_before = outgoing_lane_info[0]['end_pose'][1]
        #         start_end_position = np.append(start_end_position, [x_before, y_before, x_(after), y_(after)], axis=0)
        #         if from_ego > 100 or abs(x_(after) - ego_pose[0])>self.bbox_size  or abs(y_(after) - ego_pose[1])>self.bbox_size:
        #             return 
                
        #         plt.scatter(outgoing_lane_info[0]['start_pose'][0]-ego_pose[0]+self.bbox_size, outgoing_lane_info[0]['start_pose'][1]-ego_pose[1]+self.bbox_size, c='b', s=15)
        #         plt.scatter(x_(after)-ego_pose[0]+self.bbox_size, y_(after)-ego_pose[1]+self.bbox_size, c='b' , s=15)
        #         plot_segment(outgoing_lane[i])
        
        # patch_box = (ego_pose[0], ego_pose[1], self.bbox_size*2, self.bbox_size*2)
        # fig, ax = nusc_map.render_map_mask(patch_box, self.patch_angle, self.map_layers_list, self.canvas_size, figsize=self.fig_size, n_row=1)
                
        # for i in range(len(first_lane)):
        #     plot_segment(first_lane[i])

        # if self.show_maps:
        #     plt.show()
        #     plt.close(fig)
        # if self.save_maps:
        #     if self.train_mode:
        #         type_str = self.set + 'train'
        #     else:
        #         type_str = self.set + 'val'

        #     utils.save_maps(self, type_str, fig, idx)
        # start_end_position_output = np.resize(start_end_position, (-1,4))      

        return {
                'img'                  : img,                                                         # Type : np.array
                # 'segment'              : start_end_position_output,                                 # Type : np.array([start_x, start_y, end_x, end_y for each segment])
                # ego vehicle                                                   
                'instance_token'       : ego_instance_token,                                          # Type : str
                'sample_token'         : ego_sample_token,                                            # Type : str
                'ego_cur_pos'          : ego_pose,                                                    # Type : np.array([global_x,globa_y,global_yaw])                        | Shape : (3, )
                'ego_state'            : ego_states,                                                  # Type : np.array([vel,accel,yaw_rate]) --> local(ego's coord)          | Shape : (3, )                        | Unit : [m/s, m/s^2, rad/sec]    
                'ego_hist_num_mask'    : ego_hist_num_mask,                                           # Type : list, [self.num_past_hist, self.num_future_hist]  ... for masking history array 
                'past_global_ego_pos'  : past_poses_m,                                                # Type : np.array([global_x, global_y, global_yaw])                     | Shape : (self.num_past_hist, 3)
                'future_global_ego_pos': future_poses_m,                                              # Type : np.array([global_x, global_y, global_yaw])                     | Shape : (self.num_future_hist, 3)
                'past_cur_diff_ego'    : past_cur_time_diff_m,                                        # Difference between current and past sampling time                     | Shape : (self.num_past_hist, )  
                'future_cur_diff_ego'  : future_cur_time_diff_m,                                      # Difference between current and future sampling time                   | Shape : (self.num_future_hist, )
                # agents nearby ego (all local data in ego coordinate)
                'num_agents'           : num_agents,
                'agent_cur_pose'       : np.array(agent_local_pose_list_m),                           # Type : np.row_stack([local_x, local_y, local_yaw])                    | Shape : (self.num_max_agent, 3)
                'agent_state'          : np.array(agent_states_list_m),                               # Type : np.row_stack([vel,accel,yaw_rate]) --> local(agent's coord)    | Shape : (self.num_max_agent, 3)       | Unit : [m/s, m/s^2, rad/sec]     
                # 'agent_past_pose'      : np.array(agent_past_local_poses_list_m),                     # Type : np.row_stack([local_x, local_y, local_yaw])                    | Shape : (self.num_max_agent, self.num_past_hist, 3)
                # 'agent_future_pose'    : np.array(agent_future_local_poses_list_m),                   # Type : np.row_stack([local_x, local_y, local_yaw])                    | Shape : (self.num_max_agent, self.num_future_hist, 3)
                'num_agent_past_hist'  : np.array(num_agent_past_hist_m),                             # Type : np.array([len_past_history for each agent])                    | Shape : (self.num_past_hist, )    
                'num_agent_future_hist': np.array(num_agent_future_hist_m)                            # Type : np.array([len_future_history for each agent])                  | Shape : (self.num_future_hist, )  
                }

        # When {vel, accel, yaw_rate} is nan, it will be shown as 0 
        # History List of records.  The rows decrease with time, i.e the last row occurs the farthest in the past.

    

if __name__ == "__main__":
    dataset = NuSceneDataset(train_mode=True)
    for i in range(dataset.__len__()):
        dataset.__getitem__(i)

    # from torch.utils.data.dataloader import DataLoader
    # from dataset import NuSceneDataset

    # dataset = NuSceneDataset(train_mode=False)

    # print(dataset.__len__())
    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # for d in dataloader:
    #     print("Cc")