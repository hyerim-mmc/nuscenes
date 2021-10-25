class Config(object):
    def __init__(self):
        self.set = 'mini'                          # 'train' or 'mini'
        self.dataset_str = 'v1.0-mini'
        self.dataset_path = '/home/hyerim/data/sets/nuscenes'
        self.device = 'cpu'
        
        # Agent history
        self.num_past_hist = 10                      
        self.num_future_hist = 10          

        # Image Processing
        self.show_imgs = False
        self.save_imgs = False
        self.img_map_layers_list = ['drivable_area', 'road_segment', 'road_block', 'lane', 'ped_crossing', 'road_divider', 'lane_divider', 'traffic_light']
        self.resolution = 0.1                     # [meters/pixel]
        self.meters_ahead = 40
        self.meters_behind = 10
        self.meters_left = 25
        self.meters_right = 25        
        
        # Agent Processing
        self.num_max_agent = 10