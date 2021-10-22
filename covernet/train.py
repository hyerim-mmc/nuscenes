import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import json
import numpy as np
import torch
import pickle

from torch import nn
from datetime import datetime
from utils import Json_Parser
from network import CoverNet, mean_pointwise_l2_distance
from torch.utils.data.dataloader import DataLoader
from dataset_covernet import NuSceneDataset_CoverNet
from nuscenes.prediction.models.backbone import ResNetBackbone
from torch.utils.tensorboard import SummaryWriter


def run(config_file):
    ###################################################### Load Config paramter ######################################################
    parser = Json_Parser(config_file)
    config = parser.load_parser()    
    device = torch.device(config['LEARNING']['device'] if torch.cuda.is_available() else 'cpu')
    lr = config['LEARNING']['lr']
    momentum = config['LEARNING']['momentum']
    n_epochs = config['LEARNING']['n_epochs']
    batch_size = config['LEARNING']['batch_size']
    val_batch_size = config['LEARNING']['val_batch_size']
    num_modes = config['LEARNING']['num_modes']
    print_size = config['LEARNING']['print_size']
    resnet_path = config['LEARNING']['weight_path']
    traj_set_path = config['LEARNING']['trajectory_set_path']
    #################################################################################################################################

    train_dataset = DataLoader(NuSceneDataset_CoverNet(train_mode=True, config_file_name=config_file), batch_size=batch_size, shuffle=True)
    val_dataset = DataLoader(NuSceneDataset_CoverNet(train_mode=False, config_file_name=config_file), batch_size=val_batch_size, shuffle=True)
    backbone = ResNetBackbone('resnet50')
    backbone.load_state_dict(torch.load(resnet_path))
    model = CoverNet(backbone, num_modes)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum) 
    criterion = nn.CrossEntropyLoss()           ## classification loss
    trajectories_set =torch.Tensor(pickle.load(open(traj_set_path, 'rb')))
    model = model.to(device)

    save_name = datetime.now().strftime("%Y%m%d-%H_%M_%S")
    writer = SummaryWriter('./result/tensorboard/' + save_name)
    net_save_path = './result/model/model_{}.pth'.format(save_name)
    writer.add_text('config', json.dumps(config))

    step = 1
    for epoch in range(n_epochs + 1):
        Loss, Val_Loss = [], []
        for data in train_dataset:
            # train_mode
            model.train()
            img_tensor = torch.Tensor(data['img']).permute(2, 0, 1).to(device)
            agent_state_vector = torch.Tensor(data['ego_state'].tolist()).to(device)
            
            prediction = model(img_tensor, agent_state_vector)
            gt = torch.Tensor(data['future_global_ego_pos'][:,:2].tolist())
            label = mean_pointwise_l2_distance(trajectories_set, gt)

            optimizer.zero_grad()
            # loss = calc_loss(prediction, label)
            loss = criterion(prediction,label)
            loss.backward()
            optimizer.step()
            step += 1

            with torch.no_grad():
                Loss.append(loss.cpu().detach().numpy())

            if step % print_size == 0:
                with torch.no_grad:
                    # eval_mode
                    model.eval()
                    k = 0
                    for val_data in val_dataset:
                        img_tensor = torch.Tensor(val_data['img']).permute(2, 0, 1).to(device)
                        agent_state_vector = torch.Tensor(val_data['ego_state'].tolist()).to(device)
                        
                        prediction = model(img_tensor, agent_state_vector)
                        gt = torch.Tensor(data['future_global_ego_pos'][:,:2].tolist())
                        label = mean_pointwise_l2_distance(trajectories_set, gt)

                        val_loss = criterion(prediction,label)
                        Val_Loss.append(val_loss.detach().cpu().numpy())
                        k += 1
                        if(k == 10):
                            break
                        
                        loss = np.array(Loss).mean()
                        val_loss = np.array(Val_Loss).mean()

                        writer.add_scalar('Loss', loss, epoch)
                        writer.add_scalar('Val Loss', val_loss, epoch)

                        print("Epoch: {}/{} | Step: {} | Loss: {:.5f} | Val_Loss: {:.5f}".format(
                                epoch + 1, n_epochs, step, loss, val_loss))
                        torch.save(model.state_dict(), net_save_path)

if __name__ == "__main__":
    # run(config_file='./covernet/covernet_config.json')
    config_file='./covernet/covernet_config.json'
    parser = Json_Parser(config_file)
    config = parser.load_parser()    
    train_dataset = NuSceneDataset_CoverNet(train_mode=False, config_file_name=config_file)
    print(train_dataset.__len__())
    dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    for data in dataloader:
        print(np.shape(data['img']))
        print(data.keys())
        break
