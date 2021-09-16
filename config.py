class Config(object):
    """
    * Annotation freq = 2 Hz
    * Dataset __getitem__() output format :
        agent_states = {'instance_token'     : instance_token_img,
                        'sample_token'       : sample_token_img,
                        'img'                : img,
                        'type'               : agent_type,
                        'pos'                : current_gloabl_pose,
                        'vel'                : current_velocity,
                        'accel'              : current_acceleration,
                        'yaw_rate'           : current_yaw_rate,
                        'past_local_pos'     : past_local_poses,
                        'future_local_pos'   : future_local_poses,
                        'past_cur_diff'      : past_cur_time_diff,
                        'future_cur_diff'    : future_cur_time_diff }
                                                                            # vel(m/s), accel(m/s^2), yaw_rate(rad/sec)
        When {pos, vel, accel} is nan, it will be shown as 0 
    """

    def __init__(self):
        self.set = 'mini'
        self.dataset_str = 'v1.0-mini'

        self.device = 'cpu'
        self.dataset_path = 'data/sets/nuscenes'

        # for history
        self.past_seconds = 6
        self.future_seconds = 6

        # Image Processing
        self.show_maps = True
        self.save_maps = False
        self.rasterized = True
        self.raster_size = [224,224]
        self.pixel_size = [0.5, 0.5]
        self.ego_center = [0.25, 0.5]

        # Map Processing
        self.map_layers_list = ['drivable_area', 'road_segment', 'road_block',
                                'lane', 'ped_crossing', 'walkway', 'stop_line',
                                'carpark_area', 'road_divider', 'lane_divider', 'traffic_light']
        self.color_list = [(255, 255, 255), (255, 255, 255), (255, 255, 255),
                            (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                            (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255)]

        self.resolution = 0.1                       # meters/pixel
        self.meters_ahead = 40
        self.meters_behind = 10
        self.meters_left = 25
        self.meters_right = 25        
        self.patch_box = (300, 1700, 100, 100)
        self.patch_angle = 0                       # orientation where North is up
        self.canvas_size = (224,224)


