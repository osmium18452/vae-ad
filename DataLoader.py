import pickle

import numpy as np
from matplotlib import pyplot as plt


class DataLoader:
    def __init__(self, train_file, test_file, label_file, normalize=False, n_variate=None):
        f = open(train_file, 'rb')
        self.train_data = pickle.load(f)
        print(self.train_data[0])
        # print(self.train_data.shape)
        f.close()
        f = open(test_file, 'rb')
        self.test_data = pickle.load(f)
        # print(self.test_data.shape)
        f.close()
        f = open(label_file, 'rb')
        self.label_data = pickle.load(f)
        f.close()

        self.zero_variate = np.where(np.sum(self.train_data, axis=0) == 0)[0]
        self.non_zero_variate = np.where(np.sum(self.train_data, axis=0) != 0)[0]
        self.non_zero_data_train = self.train_data.transpose()[self.non_zero_variate[:n_variate]].transpose()
        self.non_zero_data_test = self.test_data.transpose()[self.non_zero_variate[:n_variate]].transpose()
        # normalization
        self.train_std = np.std(self.non_zero_data_train, axis=0)
        self.train_mean = np.mean(self.non_zero_data_train, axis=0)
        # print(self.non_zero_data_train[0])
        if normalize:
            self.non_zero_data_train = (self.non_zero_data_train - self.train_mean) / self.train_std
            self.non_zero_data_test = (self.non_zero_data_test - self.train_mean) / self.train_std
        # print(self.train_std,self.train_mean,self.non_zero_data_train[0],sep='\n')

    def prepare_vae_data_set(self, T=1):
        self.vae_train_set = []
        self.vae_test_set = []
        train_samples = self.non_zero_data_train.shape[0] - T + 1
        test_samples = self.non_zero_data_test.shape[0] - T + 1
        for i in range(train_samples):
            self.vae_train_set.append(self.non_zero_data_train[i:i + T])
        for i in range(test_samples):
            self.vae_test_set.append(self.non_zero_data_test[i:i + T])
        self.vae_train_set = np.array(self.vae_train_set)
        self.vae_test_set = np.array(self.vae_test_set)

    def load_vae_train_set(self):
        return self.vae_train_set

    def load_vae_test_set(self):
        return self.vae_test_set

    def load_ground_truth(self):
        return self.label_data

    def load_anomaly_position(self):
        return np.nonzero(self.label_data)[0]


if __name__ == '__main__':
    # data=np.load('sector1.npy')
    # f=open('sector1.pkl','wb')
    # pickle.dump(data,f)
    # f.close()
    # train_file = 'machine-2-1.train.pkl'
    # test_file = 'machine-2-1.test.pkl'
    train_file = 'machine-2-1.train.pkl'
    test_file = 'machine-2-1.test.pkl'
    label_file = 'machine-2-1.label.pkl'
    graph_file = 'machine-2-1.camap.pkl'
    dataloader = DataLoader(train_file, test_file, label_file, normalize=True)
    dataloader.prepare_vae_data_set()
    train_data = dataloader.load_vae_train_set()
    test_data = dataloader.load_vae_test_set()
    label_data = dataloader.load_ground_truth()
    print(train_data.shape)
    print(test_data.shape)
    print(np.nonzero(label_data)[0])
    plt.figure()
    plt.close()
    a = np.array([1, 1, 1, 0, 0, 0])
    b = np.array([1, 0, 0, 1, 0, 0])
    print(a == 1, b == 1,sep='\n')
    print((a==1)&(b==1))
