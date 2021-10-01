from nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion.quaternion import Quaternion
from nuscenes.prediction import PredictHelper

import matplotlib.pyplot as plt



DATAROOT = './data/sets/nuscenes'
nusc = NuScenes('v1.0-mini', dataroot=DATAROOT)
dataset = get_prediction_challenge_split("mini_train", dataroot=DATAROOT)

helper = PredictHelper(nusc)
instance_token, sample_token = dataset[0].split("_")

sample_annotation = helper.get_sample_annotation(instance_token, sample_token)

my_scene = nusc.scene[0]
# print("scene 첫번째!!! \n", my_scene['name'])

first_sample_token = my_scene['first_sample_token']
LIDAR_sample = nusc.get('sample', first_sample_token)['data']['LIDAR_TOP']
ego_position = nusc.get('ego_pose', LIDAR_sample)['translation']
print("ego position 위치 = ", ego_position)

nusc_map = NuScenesMap(map_name='singapore-onenorth', dataroot=DATAROOT)

# ego_plot = nusc_map.render_egoposes_on_fancy_map(nusc, scene_tokens=[nusc.scene[0]['token']], verbose=False, render_legend=True)
ego_plot = nusc_map.render_egoposes_on_fancy_map(nusc, scene_tokens=my_scene['token'], verbose=False, render_legend=True)

plt.scatter([ego_position[0]], [ego_position[1]], c='red' , s=100)

# closest_lane = nusc_map.get_closest_lane(ego_position[0], ego_position[1], radius=2)

# lane_record = nusc_map.get_arcline_path(closest_lane)
# print("start position : ", lane_record[0]['start_pose']) # x, y, yaw
# print("end position : ", lane_record[0]['end_pose'])
# print("\n")
# # 왜 중간중간 비는 구간이 생기는지 파악해야함!!!

# # lane_before = nusc_map._get_connected_lanes(closest_lane, 'incoming')
# # lane_record = nusc_map.get_arcline_path(lane_before[0])
# # print("start position : ", lane_record[0]['start_pose'])
# # print("end position : ", lane_record[0]['end_pose'])

# lane_after = nusc_map._get_connected_lanes(closest_lane, 'outgoing')[0]

# i = 0

# ###### recursive function 선언하기!!!
# # def plot_segment(lane_token):
# #     out_case_number = len(nusc_map._get_connected_lanes(lane_token, 'outgoing')
    
# #     for i in range(out_case_number):
# #         lane_token = nusc_map.get_arcline_path(lane_temp_outgoing[i])
# #         plot_segment(lane_token)


# while(1):

#     if i==11:    # boundary 밖에 나가는 조건으로 바꾸기
#         break
#     lane_temp_outgoing = nusc_map._get_connected_lanes(lane_after, 'outgoing')  # 얘는 바깥으로 나가는 경로의 token 값을 저장하는 용도, 두갈래길이면 토큰 두개 저장
#     # print("lane_temp_outgoing = ", lane_temp_outgoing)

#     if len(lane_temp_outgoing) == 1:
#         print(f"{i+1}번째 lane ")
#         lane_record = nusc_map.get_arcline_path(lane_temp_outgoing[0])  # 얘는 내가 얻고자 하는 start_pose, end_pose의 정보를 가지고 있음
#         # print("lane_record : ", lane_record)
#         print("start position : ", lane_record[0]['start_pose'])
#         print("end position : ", lane_record[0]['end_pose'])
#         plt.scatter(lane_record[0]['start_pose'][0], lane_record[0]['start_pose'][1], c='b' , s=100)
#         plt.scatter(lane_record[0]['end_pose'][0], lane_record[0]['end_pose'][1], c='b' , s=100)
#         lane_after = lane_temp_outgoing[0]
#         print("\n")
    
#     # 갈림길이 나왔을 때 각각의 경로로 반복문이 돌아야함!!!
    
#     # for j in range(len(lane_temp_outgoing)):
#     #     print(f"{i+1}번째 중 {j+1}번째 갈림길")
#     #     print("\n")
#     i = i+1

# # # camera로 현재 scene 보는 용도
# # my_scene_token = nusc.field2token('scene', 'name', my_scene['name'])[0]
# # nusc.render_scene_channel(my_scene_token, 'CAM_FRONT')  # 동영상 재생


# # plt.savefig("plot", bbox_inches = 'tight')
# print("@@@@@@@@@@@@@@@@@ 종료 @@@@@@@@@@@@@@@")