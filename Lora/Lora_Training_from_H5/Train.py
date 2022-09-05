import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from torchaudio.transforms import Spectrogram

from Lora.Lora_Training_from_H5 import Lora_Model
from Lora.Lora_Dataset import Lora_H5_dataset
from Lora.Lora_Transforms import convert_to_complex_H5


from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer

import yaml

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)

if __name__ == '__main__':

    with open(r'config.yaml') as f:
        config = yaml.load(stream=f, Loader=yaml.FullLoader)
    if config is not None:


        data_src = config['DATA']['PATH']



    path= data_src

    get_complex_from_H5 = convert_to_complex_H5()
    transform_spec=Spectrogram()
    training_dataset = Lora_H5_dataset(path, convert_complex=False, transform=transform_spec, data_start=None,
                                                  data_end=None,is_train=False)
    train_loader=DataLoader(training_dataset, shuffle=True,num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        print(torch.cuda.get_device_name())
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Training on GPU!')
    embedding_dims = 32
    batch_size = 4
    epochs = 50

    model = Lora_Model.Lora_Network(emb_dim=embedding_dims,input_bins=1)

    model.apply(init_weights)

    #model = torch.jit.script(model).to(device)
    model =  model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #criterion = torch.jit.script(Lora_Model.TripletLoss())

    model.train()

    miner = miners.MultiSimilarityMiner()
    loss_func = losses.TripletMarginLoss(distance = CosineSimilarity(),reducer = ThresholdReducer(high=0.3),
                                    embedding_regularizer = LpRegularizer())

    for epoch in tqdm(range(epochs), desc="Epochs"):
        running_loss = []
        """for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(
                tqdm(train_loader, desc="Training", leave=False)):
            anchor_img = anchor_img.to(device)
            positive_img = positive_img.to(device)
            negative_img = negative_img.to(device)

            optimizer.zero_grad()
            anchor_out = model(anchor_img)
            positive_out = model(positive_img)
            negative_out = model(negative_img)

            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()"""

        for i, (data,labels) in enumerate(train_loader):
            data = data.unsqueeze(1)
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            #print("shape of data before model : ",data.shape)
            #data=data.permute(1, 2, 0)
            #print("shape of data before model : ", data.shape)
            embeddings = model(data)
            #print("labels shape : ",labels.ndim,labels.shape)
            #print("after label change", labels.size())
            labels=torch.flatten(labels)
            #labels=labels.squeeze(1)

            #logits=loss_func.get_logits(embeddings)
            #print("logits : ",logits)
            hard_pairs = miner(embeddings, labels)
            #print("labels shape : ", labels.ndim, labels.shape)
            loss = loss_func(embeddings, labels,hard_pairs)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.cpu().detach().numpy())
        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, epochs, np.mean(running_loss)))


    PATH=r'D:\git-repos\Signal-Classifier\Lora\Lora_Training_from_H5\Lora_m'
    torch.save(model.state_dict(), PATH)