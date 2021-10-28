# Nuscenes Dataset processing
This repository contains Nuscenes dataset processing.

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


## Run
Please check DATASET_PATH in advance!


Check more details about config and dataset output format in ```instruction.md```


Choose Dataset Processing type from ```dataset.py``` or ```dataset_mini.py```
    - ```dataset.py``` samples img, ego_vehicle_state, ego past/future history, agent data etc. 
    - ```dataset_mini/dataset_mini.py``` samples img, ego_vehicle_state, ego_future_history


    - Write own parsing configuration ```config.py``` or ```dataset_mini/mini_config.json```
    - Use ```python dataset.py``` or ```dataset_mini/dataset_mini.py``` for Dataloader
    