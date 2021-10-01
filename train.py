import torch

def loss():
    pass

def run()):
    optimizer = torch.optim.SGD(covernet.parameters(), lr=1e-2, momentum=0.9) 
    # epoch


    nb_epochs = 1000
    for epoch in range(nb_epochs + 1):
        for batch_idx, samples in enumerate(dataloader):
            if batch_idx == 92:
                break
            x_train, y_train = samples
            sample_list = x_train[0]
            instance_list = x_train[1]
            try:
                img_tensors = get_train_img(sample_list, instance_list,input_representation)
            except:
                continue
            x_train[2] = torch.squeeze(x_train[2],1)
            prediction = covernet(img_tensors, x_train[2].to(device)).to(device)
            #print(y_train)
            loss = compute_loss(prediction, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    run()