import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from net import Net
from dataset import Dataset

class Solver():
    def __init__(self, args):
        self.args = args
        # prepare a datasets
        self.train_data = Dataset(data_root=args.data_root train=True)
        self.test_data  = Dataset(data_root=args.data_root, train=False)
        
        #initialize loss function, optimizer, network, data loader
        self.train_loader = DataLoader(dataset=self.train_data,
                                       batch_size=args.batch_size,
                                       num_workers=4,
                                       shuffle=True, drop_last=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Net(type=args.type).to(self.device)
        self.loss_fn = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        
        self.checkpoint_path = os.path.join(os.getcwd(), 'checkpoint')

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        
    def fit(self):
        #TODO
        args = self.args

        for epoch in range(args.max_epochs):
            self.net.train()
            for step, inputs in enumerate(self.train_loader):
                images = inputs[1].to(self.device)
                labels = inputs[0].to(self.device)
                pred = self.net(images)
                loss = self.loss_fn(pred, labels)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                train_acc = self.evaluate(self.train_data)
                test_acc = self.evaluate(self.test_data)

                print("Epoch [{}/{}] Loss: {:.3f} Train Acc: {:.3f}, Test Acc: {:.3f}".
                      format(epoch + 1, args.max_epochs, loss.item(), train_acc, test_acc))
                #print("Epoch [{}/{}] Loss: {:.3f} ".
                #      format(epoch + 1, args.max_epochs, loss.item()))
            

    def evaluate(self, global_step):
        #TODO
        
        
    def save(self, ckpt_name, global_step):
        save_path = os.path.join(
            self.checkpoint_path, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)
