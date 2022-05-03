import os
import time
import torch
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from collections import defaultdict


import numpy as np

from datasets import learningSampleDataset
from models import VAE
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(
        recon_x.view(-1, 64*64), x.view(-1, 64*64), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return (BCE + KLD) / x.size(0)

def main():
    epochs=1000
    encoder_sizes=[4096, 2048]
    decoder_sizes=[2048,4096]
    latent_size=15
    learning_rate=0.01
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = time.time()


    
    dataset = learningSampleDataset(type="train",goal="puck")
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    dataset = learningSampleDataset(type="test",goal="puck")
    data_loader_test = DataLoader(dataset, batch_size=64, shuffle=False)

    x_test,c1_test, c2_test,c3_test,c4_test=next(iter(data_loader_test))

    x_test, c1_test,c2_test,c3_test,c4_test = x_test.float().to(device), c1_test.to(device),c2_test.to(device),c3_test.to(device),c4_test.to(device)

    import datetime
    datetime_now=str(datetime.datetime.now().time()).replace(".", "")
    datetime_now=datetime_now.replace(":","")
    os.makedirs('figs/'+datetime_now,exist_ok=True)

    
    vae = VAE(
        encoder_layer_sizes=encoder_sizes,
        latent_size=latent_size,
        decoder_layer_sizes=decoder_sizes,
        num_labels=64 ).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

    logs = defaultdict(list)

    for epoch in range(epochs):

        for iteration, (x,c1, c2,c3,c4) in enumerate(data_loader):

            x, c_1,c_2,c_3,c_4 = x.float().to(device), c1.to(device),c2.to(device),c3.to(device),c4.to(device)


            recon_x, mean, log_var, z = vae(x, c_1,c_2,c_3,c_4)

            loss = loss_fn(recon_x, x, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

            if iteration % 100 == 0 or iteration == len(data_loader)-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, epochs, iteration, len(data_loader)-1, loss.item()))



                z = torch.randn([c1_test.size(0), latent_size]).to(device)
                c1_test=c1_test.reshape(-1,1)
                c2_test=c2_test.reshape(-1,1)
                c3_test=c3_test.reshape(-1,1)
                c4_test=c4_test.reshape(-1,1)
                x = vae.inference(z, c1=c1_test,c2=c2_test,c3=c3_test,c4=c4_test)


                plt.figure()
                plt.figure(figsize=(30.0, 30.0))
                for p in range(10):
                    ax=plt.subplot(5, 2, p+1)
                    ax.set_xlim((0, 64))
                    ax.set_ylim((64, 0))
                    ax.imshow(x[p].view(64, 64).cpu().data.numpy())
                    ax.plot(c2_test[p].item(),c1_test[p].item(),'rp',markersize = 10,markerfacecolor="w")
                    ax.plot(c4_test[p].item(),c3_test[p].item(),'o',markersize = 10,markerfacecolor="w")


            

                plt.savefig('figs/'+datetime_now+'/Epoch{:d}Iter{:d}.png'.format(epoch, iteration),dpi=300)
                plt.clf()
                plt.close('all')


if __name__ == '__main__':
    main()
