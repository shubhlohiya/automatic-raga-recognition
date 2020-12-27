import torch
from torch.utils.data import TensorDataset, DataLoader
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_trainloader(X_train, Y_train, batch_size = 40):
    """
    :param X_train: Training data as a torch tensor
    :param Y_train: Training data as a torch tensor
    :param batch_size: required batch size while training
    :return: Dataloader object
    """

    trainset = TensorDataset(X_train, Y_train)
    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)

    return trainloader

def trainer(trainloader, model, optimizer, criterion, epochs, device=device):
    """

    :param trainloader: Dataloader Object
    :param model: instance of our deep learning model
    :param optimizer: optimizer to be used to update weights
    :param criterion: loss function to be used
    :param epochs: number of epochs to train for
    :return: losslist of average losses per epoch
    """
    model.train()
    losslist = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 15 == 14:  # print every 15 mini-batches
                print(f'Epoch: {epoch+1} | Batches Done: {i + 1}/450 | '
                      f'Loss: {(running_loss/15):.3f}')
                epoch_loss += running_loss
                running_loss = 0.0

        losslist.append(epoch_loss / len(trainloader))
        print("=" * 50)
        print(f"EPOCH {epoch+1} OVERALL LOSS: {losslist[-1]:.3f}")
        print("=" * 50)
        if epoch % 5 == 4:
            path = f"../models/lstm_{epoch+1}_checkpoint.pth"
            print(f"Saving model weights at Epoch {epoch+1} ...")
            torch.save(model.state_dict(), path)

    return losslist