import pickle
from typing import List, Tuple
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import tqdm
import numpy as np

from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

nuscenes = NuScenes('v1.0-mini', dataroot='/home/hyerim/data/sets/nuscenes')
nusc_map = NuScenesMap(dataroot='/home/hyerim/data/sets/nuscenes', map_name='singapore-onenorth')

with open("records_in_patch.pickle","rb") as fr:
    data = pickle.load(fr)
# print(data)

non_geometric_polygon_layers = ['drivable_area', 'road_segment', 'road_block', 'lane', 'ped_crossing',
                                             'walkway', 'stop_line', 'carpark_area']

def render_map_patch(
                        box_coords: Tuple[float, float, float, float],
                        layer_names: List[str] = None,
                        alpha: float = 0.5,
                        figsize: Tuple[float, float] = (15, 15),
                        render_egoposes_range: bool = True,
                        render_legend: bool = True):
    """
    Renders a rectangular patch specified by `box_coords`. By default renders all layers.
    :param box_coords: The rectangular patch coordinates (x_min, y_min, x_max, y_max).
    :param layer_names: All the non geometric layers that we want to render.
    :param alpha: The opacity of each layer.
    :param figsize: Size of the whole figure.
    :param render_egoposes_range: Whether to render a rectangle around all ego poses.
    :param render_legend: Whether to render the legend of map layers.
    :param bitmap: Optional BitMap object to render below the other map layers.
    :return: The matplotlib figure and axes of the rendered layers.
    """
    x_min, y_min, x_max, y_max = box_coords

    if layer_names is None:
        layer_names = non_geometric_polygon_layers

    fig = plt.figure(figsize=figsize)

    local_width = x_max - x_min
    local_height = y_max - y_min
    assert local_height > 0, 'Error: Map patch has 0 height!'
    local_aspect_ratio = local_width / local_height

    ax = fig.add_axes([0, 0, 1, 1 / local_aspect_ratio])

    for layer_name in layer_names:
        pass
        # _render_layer(ax, layer_name, alpha)

    x_margin = np.minimum(local_width / 4, 50)
    y_margin = np.minimum(local_height / 4, 10)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    if render_egoposes_range:
        ax.add_patch(Rectangle((x_min, y_min), local_width, local_height, fill=False, linestyle='-.', color='red',
                                lw=2))
        ax.text(x_min + local_width / 100, y_min + local_height / 2, "%g m" % local_height,
                fontsize=14, weight='bold')
        ax.text(x_min + local_width / 2, y_min + local_height / 100, "%g m" % local_width,
                fontsize=14, weight='bold')

    if render_legend:
        ax.legend(frameon=True, loc='upper right')

    return fig, ax


# Settings
patch_margin = 2
min_diff_patch = 30

# Get logs by location.
log_location = 'singapore-onenorth'
log_tokens = [log['token'] for log in nuscenes.log if log['location'] == log_location]
assert len(log_tokens) > 0, 'Error: This split has 0 scenes for location %s!' % log_location

# Filter scenes.
scene_tokens_location = [e['token'] for e in nuscenes.scene if e['log_token'] in log_tokens]
assert len(scene_tokens_location) > 0, 'Error: Found 0 valid scenes for location %s!' % log_location

map_poses = []
for scene_token in scene_tokens_location:
    # Check that the scene is from the correct location.
    scene_record = nuscenes.get('scene', scene_token)
    scene_name = scene_record['name']
    scene_id = int(scene_name.replace('scene-', ''))     
    log_record = nuscenes.get('log', scene_record['log_token'])
    assert log_record['location'] == log_location, \
        'Error: The provided scene_tokens do not correspond to the provided map location!'

    # For each sample in the scene, store the ego pose.
    sample_tokens = nuscenes.field2token('sample', 'scene_token', scene_token)
    for sample_token in sample_tokens:
        sample_record = nuscenes.get('sample', sample_token)

        # Poses are associated with the sample_data. Here we use the lidar sample_data.
        sample_data_record = nuscenes.get('sample_data', sample_record['data']['LIDAR_TOP'])
        pose_record = nuscenes.get('ego_pose', sample_data_record['ego_pose_token'])

        # Calculate the pose on the map and append.
        map_poses.append(pose_record['translation'])

# Check that ego poses aren't empty.
assert len(map_poses) > 0, 'Error: Found 0 ego poses. Please check the inputs.'

# Compute number of close ego poses.
map_poses = np.vstack(map_poses)[:, :2]

# Render the map patch with the current ego poses.
min_patch = np.floor(map_poses.min(axis=0) - patch_margin)
max_patch = np.ceil(map_poses.max(axis=0) + patch_margin)
diff_patch = max_patch - min_patch
if any(diff_patch < min_diff_patch):
    center_patch = (min_patch + max_patch) / 2
    diff_patch = np.maximum(diff_patch, min_diff_patch)
    min_patch = center_patch - diff_patch / 2
    max_patch = center_patch + diff_patch / 2
my_patch = (min_patch[0], min_patch[1], max_patch[0], max_patch[1])
fig, ax = render_map_patch(my_patch, non_geometric_polygon_layers, figsize=(10, 10),
                                render_egoposes_range=True,
                                render_legend=True)

# Plot in the same axis as the map.
# Make sure these are plotted "on top".
ax.scatter(map_poses[:, 0], map_poses[:, 1], s=20, c='k', alpha=1.0, zorder=2)
plt.axis('off')

plt.savefig('out.png', bbox_inches='tight', pad_inches=0)
