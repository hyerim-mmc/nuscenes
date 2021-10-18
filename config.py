class Config(object):
    def __init__(self):
        self.set = 'mini'           # 'train' or 'mini'
        self.dataset_str = 'v1.0-mini'
        self.dataset_path = '/home/hyerim/data/sets/nuscenes'
        self.device = 'cpu'
        
        # Agent history
        self.past_seconds = 6/2                      
        self.future_seconds = 6/2

        # Image Processing
        self.show_imgs = False
        self.save_imgs = False
        # self.raster_size = [224,224]        # doesn't work
        # self.pixel_size = [0.5, 0.5]        # doesn't work
        # self.ego_center = [0.25, 0.5]       # doesn't work
        self.img_map_layers_list = ['drivable_area', 'road_segment', 'road_block', 'lane', 'ped_crossing', 'road_divider', 'lane_divider', 'traffic_light']

        # Map Processing
        self.show_maps = False
        self.save_maps = False
        self.map_layers_list = ['lane_divider']   # choose from ['drivable_area', 'road_segment', 'road_block', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area', 'road_divider', 'lane_divider', 'traffic_light']
        self.lane_type = 'incoming'               # 'incoming' or 'outgoing'
        self.canvas_size = (200, 200)
        self.fig_size= (8,4)
        self.bbox_size_limit = 100                # [meter]
        self.resolution = 0.1                     # [meters/pixel]
        self.meters_ahead = 40
        self.meters_behind = 10
        self.meters_left = 25
        self.meters_right = 25        
        self.patch_angle = 0                      # orientation where North is up
        
        # Agent Processing
        self.num_max_agent = 10

        # Mask (Use when you get data in specific region)
        self.mask = [300,1250,500,1000]