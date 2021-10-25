## config
1) self.set = 'mini'            
--> Set nuscene dataset type. Choose from 'train','mini'

2) self.dataset_str = 'v1.0-mini'
--> Set your download dataset type from mini, trainval, test

3) self.dataset_path = '/home/hyerim/data/sets/nuscenes'
--> Absolute dataset path

4) self.device = 'cpu'
--> Set device when learning (cpu or cuda)

5) self.num_past_hist = 10       
--> Set how many path past history to collect (data publish each 0.5 sec)
	ex) When you set this to 10, you will get 5 sec of data

6) self.num_future_hist = 10
--> Set how many path future history to collect (data publish each 0.5 sec)

7) self.show_imgs = False
--> Set load rasterized image or not

8) self.save_imgs = False
--> Set save rasteried image in dataset folder or not

9) self.img_map_layers_list
--> Set wanted map layer to rasterize image
--> Collect from ['drivable_area', 'road_segment', 'road_block', 'lane', 
		'ped_crossing', 'walkway', 'stop_line', 'carpark_area', 
		'road_divider', 'lane_divider', 'traffic_light']

10) self.resolution = 0.1                     # [meters/pixel]

11) self.meters_ahead = 40
12) self.meters_behind = 10
13) self.meters_left = 25
14) self.meters_right = 25        
--> Variables when choosing agent or rasterizing w.r.t ego coordinate

15) self.num_max_agent = 10
--> Set the number of agents when collecting agent data from ego vehicle coordinate 


## dataset output
1) img 
	: rasterized image 

2) ego_cur_pos 
	: current ego position data, {global x, y, yaw}
		type = np.ndarray, shape = (3,)
3) ego_state 
	: current ego vehicle state data, {vel, accel, yaw_rate}
		type = np.ndarray, shape = (3,)
4) num_ego_hist
	: the number of ego history for masking {len(past_hist), len(future_hist)}
		type = np.ndarray, shape = (2,)
5) past_global_ego_pos 
	: past ego position history
		type = np.ndarray, shape = (self.num_past_hist, 3)
6) future_global_ego_pos 
	: future ego position history
		type = np.ndarray, shape = (self.num_future_hist, 3)

7) num_agents 
	: the number of agent data nearby ego
		type = int
8) agent_cur_pose 
	: current agent position data (transformed to ego vehicle coordinate)
		type = np.ndarray, shape = (self.num_max_agent, 3)
9) agent_state 
	: current agent vehicle state data (transformed to ego vehicle coordinate)
		type = np.ndarray, shape = (self.num_max_agent, 3)
10) agent_past_pose 
	: past agent position history {local x, y, yaw}
		type = np.ndarray,
		shape = (self.num_max_agent, self.num_past_hist, 3)
11) agent_future_pose 
	: future agent position history {local x, y, yaw}
		type = np.ndarray, 
		shape = (self.num_max_agent, self.num_future_hist, 3)
12) num_agent_past_hist 
	: the number of agent past history data 
		type = np.ndarray, shape = (self.num_past_hist,)
13) num_agent_future_hist 
	: the number of agent future history data 
		type = np.ndarray, shape = (self.num_future_hist,)