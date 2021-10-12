# Nuscenes Dataset processing & CoverNet Implementation
This repository contains an implementation of [CoverNet](https://arxiv.org/pdf/1911.10298.pdf) and Nuscenes dataset processing.

*Phan-Minh, Tung, et al. "Covernet: Multimodal behavior prediction using trajectory sets." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.*

## Setup
1. Download Repository ```git clone https://github.com/hyerim-mmc/nuscenes.git```
2. Download [Nuscenes Dataset_Full dataset(v1.0)](https://www.nuscenes.org/download) 

    - Dataset Architecture should be as follows
    ```
   ${project_folder_name}
      |__data
        |__sets
          |__nuscenes
             |__maps
             |__samples
             |__sweeps
             |__v1.0-mini
             |__detection.json
    ```

3. Download [Map expansion](https://www.nuscenes.org/download) 
   
   - Extract the contents (folders basemap, expansion and prediction) to your nuScenes maps folder. 
4. Download [Nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit#getting-started-with-nuscenes)
5. Download [Pytorch](https://pytorch.org/get-started/locally/)
6. Download Tensorboard ```pip install tensorboard``` or ```conda install tensorboard```

## Run
Please check DATASET_PATH in advance!

1. Dataset Processing
    - Write own parsing configuration ```config.py```
    - Run ```python dataset.py```
    
2. CoverNet Implementation (NOT COMPLETED)
    - Write own parsing/learning configuration ```covernet/covernet_config.json```
    - Run ```python covernet/train.py```
    - Results will be saved in ```covernet/result``` folder
