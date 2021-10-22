import argparse
import os
import paddle
import numpy as np
import paddle.nn as nn
from paddle import optimizer
from tqdm import tqdm
from paddle.io import DataLoader, random_split
from dataset.dataset import MyDataset
from model.model import Net
def get_args():
    parser = argparse.ArgumentParser(description='Train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=400,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=16,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()

def train(net, epoch, batch_size, lr, val_percent):
    save_dir = './checkpoint/'
    dataset_dir = '../dataset/'
    dataset = MyDataset(dataset_dir)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    # train, val = random_split(dataset, [n_train, n_val])
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    # val_dataloader = DataLoader(val, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    optim = optimizer.Adam(learning_rate=lr, parameters=net.parameters())

    criterion = nn.MSELoss()

    for e in range(epoch):
        net.train()
        with tqdm(total=len(dataset), desc=f'Epoch {e + 1}/{epoch}', unit='img') as p:
            for batch in train_dataloader:
                imgs, labels = batch['img'], batch['label']

                preds = net(imgs)
                loss = criterion(preds, labels)
                p.set_postfix(**{'loss (batch)': loss.item()})
                optim.clear_grad()
                loss.backward()
                optim.step()
                p.update(imgs.shape[0])
                
        if e % 10 == 0:
            paddle.save(net.state_dict(), os.path.join(save_dir, str(e)+'.pdparams'))

if __name__ == '__main__':
    
    args = get_args()
    net = Net(3)
    train(net, args.epochs, args.batchsize, args.lr, 10)
