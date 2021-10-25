import torch
import random
from nuscenes.prediction.models.covernet import CoverNet
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.models.backbone import ResNetBackbone
from nuscenes.prediction.input_representation.static_layers import draw_lanes_on_image
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from dataset import CoverNet_Dataset
import pickle
import numpy as np
import cv2
from typing import Dict, List, Tuple, Callable
from nuscenes.prediction.input_representation.utils import get_crops, get_rotation_matrix, convert_to_pixel_coords

PATH = './weights/model_mini.pt'
DATAROOT = '/home/hun/data/sets/nuscenes'
mode = 64
nuscenes = NuScenes('v1.0-mini', dataroot=DATAROOT)
helper = PredictHelper(nuscenes)
agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history=1)

backbone = ResNetBackbone('resnet50')
covernet = CoverNet(backbone, num_modes=mode)
covernet.load_state_dict(torch.load(PATH))
covernet.eval()

"""
1) backbone, covernet load
2) eval mode
3) 

"""
def trajectory2lane(trajectory):
    str = 'custom_lane'
    x = trajectory[:,0]
    y = trajectory[:,1]
    yaw = np.zeros_like(x)
    l = []
    for i,v in enumerate(x):
        t = (x[i],y[i],yaw[i])
        l.append(t)
    dic = {str:l}
    return dic

def draw_lanes_on_image(image: np.ndarray,
                        lanes: Dict[str, List[Tuple[float, float, float]]],
                        mode,
                        resolution: float):
    resolution = 1/resolution
    i = 0
    if mode =="prediction":
        color = (0, 255, 0) # green
    elif mode =="label":
        color = (255, 0, 0) # blue
    elif mode =="true":
        color = (0, 0, 255) # red

    for poses_along_lane in lanes.values():
        for start_pose, end_pose in zip(poses_along_lane[:-1], poses_along_lane[1:]):
            if i ==0:
                start_pixels = (250,410)
                end_pixels = (int(start_pose[0]*resolution + 250), int(-start_pose[1]*resolution + 410))
                cv2.line(image, start_pixels, end_pixels, color,thickness=1)
            start_pixels = (int(start_pose[0]*resolution + 250), int(-start_pose[1]*resolution + 410))
            end_pixels = (int(end_pose[0]*resolution + 250), int(-end_pose[1]*resolution + 410))
            # Need to flip the row coordinate and the column coordinate
            # because of cv2 convention
            cv2.line(image, start_pixels, end_pixels, color, thickness=3)
            i+=1
    return image

def convert_to_pixel_coords(location: Tuple[float, float],
                            center_of_image_in_global: Tuple[float, float],
                            center_of_image_in_pixels: Tuple[float, float],
                            resolution: float = 0.1) -> Tuple[int, int]:
    print("location : {}".format(location))
    print("center_of_image_in_global : {}".format(center_of_image_in_global))
    x, y = location # 659, 1613
    # center_of_image : 617, 1636
    x_offset = (x - center_of_image_in_global[0]) # 42
    y_offset = (y - center_of_image_in_global[1]) # -20
    print(x_offset)
    x_pixel = x_offset / resolution # 420�� ��.
    # Negate the y coordinate because (0, 0) is ABOVE and to the LEFT
    y_pixel = -y_offset / resolution
    print(x_pixel)
    row_pixel = int(center_of_image_in_pixels[0] + y_pixel)
    column_pixel = int(center_of_image_in_pixels[1] + x_pixel)

    return row_pixel, column_pixel

def get_all_candidate(all_trajectory):
    dict = {}
    for k, traj in enumerate(all_trajectory):
        x = traj[:,0]
        y = traj[:,1]
        yaw = np.zeros_like(x)
        l = []
        for i,v in enumerate(x):
            t = (x[i],y[i],yaw[i])
            l.append(t)
        dict[k] = l
    return dict

trajectories = pickle.load(open('epsilon_8.pkl', 'rb'))
trajectories = torch.Tensor(trajectories)
all_traj = get_all_candidate(trajectories)

dataset = CoverNet_Dataset(DATAROOT, helper, verbose='False', mode='val')
input, label, instance_token, sample_token = dataset.get_val(8)

img_agent = agent_rasterizer.make_representation(instance_token, sample_token)
agent_box = agent_rasterizer.agent_box

state = input[1]
img = input[0]
img = torch.unsqueeze(img,0)
state.unsqueeze(0)
state = torch.squeeze(state,1)

logits = covernet(img, state)
logits = logits.detach().numpy()
index = np.argmax(logits)
pred_trajectory = trajectories[index]
ground_truth = trajectories[label]
pred_trajectory = pred_trajectory.detach().numpy()
ground_truth = ground_truth.detach().numpy()

future_xy_local = helper.get_future_for_agent(instance_token, sample_token, seconds=6, in_agent_frame=True)
annotation = helper.get_sample_annotation(instance_token, sample_token)

trans = annotation['translation']

my_sample = nuscenes.get('sample', sample_token)
sensor = 'CAM_BACK'
cam_front_data = nuscenes.get('sample_data', my_sample['data'][sensor])
ego_pose = nuscenes.get('ego_pose', cam_front_data['ego_pose_token'])
ego_pose = ego_pose['translation']

print("agent xy gloabal : {}".format(trans))
print("ego xy gloabal : {}".format(ego_pose))

agent_pixels = (int(500 / 2), int(500 / 2))
pred_lane = trajectory2lane(pred_trajectory) 
true_lane = trajectory2lane(ground_truth)
real_lane = trajectory2lane(future_xy_local)

img = torch.squeeze(img,0)
img = torch.Tensor(img).permute(1, 2, 0)
img = img.detach().numpy()
img_2 = np.copy(img)

empty_img = np.zeros((500,500,3))


result_img = draw_lanes_on_image(img, pred_lane, 'prediction', 0.1)
result_img = draw_lanes_on_image(img, true_lane, 'label', 0.1)
result_img = draw_lanes_on_image(img, real_lane, 'true', 0.1)
cv2.imshow("result_img",result_img)

all_traject_img = draw_lanes_on_image(img_2, all_traj, 'prediction', 0.1)
cv2.imshow("all",all_traject_img)
cv2.waitKey(0)
cv2.destroyAllWindows