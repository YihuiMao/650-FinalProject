from torch.utils.data import Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class learningSampleDataset(Dataset):
    def __init__(self, type="train",goal="puck"):
        #self.samples = list(range(low, high))
        self.type=type
        if type=="train" and goal=="puck":
            self.my_data = np.genfromtxt('image_puck_train.csv', delimiter=',').reshape(10000,64,64)
            self.my_data_coord = np.genfromtxt('coord_puck_train.csv', delimiter=',')
            self.my_data_coord=self.my_data_coord.reshape(10000*2,2)
            self.my_data_coord=self.convert_coord2Index(self.my_data_coord).reshape(10000,2,2)
        if type=="test" and goal=="puck":
            self.my_data = np.genfromtxt('image_puck_test.csv', delimiter=',').reshape(100,64,64)
            self.my_data_coord = np.genfromtxt('coord_puck_test.csv', delimiter=',')
            self.my_data_coord=self.my_data_coord.reshape(100*2,2)
            self.my_data_coord=self.convert_coord2Index(self.my_data_coord).reshape(100,2,2)

    def __len__(self):
        if self.type=="train":
            return 10000
        else:
            return 100

    def __getitem__(self, idx):
        return self.my_data[idx],self.my_data_coord[idx][0][0],self.my_data_coord[idx][0][1],self.my_data_coord[idx][1][0],self.my_data_coord[idx][1][1]


    def convert_coord2Index(self,coord):
        index = np.floor(coord).astype(np.int64)
        return index
if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = learningSampleDataset(type="test",goal="puck")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    x,c1, c2,c3,c4=next(iter(dataloader))

    print(x.shape,c1.shape)
    for i in range(10):
        plt.figure(figsize=(8.0, 8.0))
        plt.xlim((0, 64))
        plt.ylim((0, 64))
        #print(x[].shape,c1.shape)
        plt.imshow(x[i])
        # plt.plot(my_data_coord_1[i][0][0],my_data_coord_1[i][0][1],'rp',markersize = 14)
        # plt.plot(my_data_coord_1[i][1][0],my_data_coord_1[i][1][1],'o',markersize = 14)
        plt.plot(c1[i],c2[i],'rp',markersize = 20,alpha=0.3)
        plt.plot(c3[i],c4[i],'o',markersize = 20,alpha=0.3)
        plt.show()
    # # for iteration, (x, c1,c2,c3,c4) in enumerate(dataloader):
    #     plt.figure(figsize=(8.0, 8.0))
    #     plt.xlim((0, 64))
    #     plt.ylim((0, 64))
    #     #print(x[].shape,c1.shape)
    #     plt.imshow(x[0])
    #     # plt.plot(my_data_coord_1[i][0][0],my_data_coord_1[i][0][1],'rp',markersize = 14)
    #     # plt.plot(my_data_coord_1[i][1][0],my_data_coord_1[i][1][1],'o',markersize = 14)
    #     plt.plot(c1[0],c2[0],'rp',markersize = 20,alpha=0.3)
    #     plt.plot(c3[0],c4[0],'o',markersize = 20,alpha=0.3)
    #     plt.show()
    #print(next(iter(dataloader)))