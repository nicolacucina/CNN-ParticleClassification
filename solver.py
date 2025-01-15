import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from net import Net, Net1
from dataset import Dataset

class Solver():
    def __init__(self, args):
        self.args = args
        # Set seeds for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        self.train_data = Dataset(data_root=args.data_root, scaled=args.data_scaled, type='train', seed=args.seed)
        self.val_data  = Dataset(data_root=args.data_root, scaled=args.data_scaled, type='val', seed=args.seed)
        
        #initialize data loader, loss function, optimizer, network
        self.train_loader = DataLoader(dataset=self.train_data,
                                       batch_size=args.batch_size,
                                       num_workers=1,
                                       shuffle=False, drop_last=True)

        self.val_loader = DataLoader(dataset=self.val_data,
                                      batch_size=args.batch_size,
                                      num_workers=1,
                                      shuffle=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.net = Net(type=args.type).to(self.device)
        self.net = Net1().to(self.device)
        # self.loss_fn = torch.nn.MSELoss()
        # self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_fn = torch.nn.BCEWithLogitsLoss() # combines a Sigmoid layer and the BCELoss in one single class

        self.optim = torch.optim.SGD(self.net.parameters(), lr=args.lr, weight_decay=args.decay)     
        # self.optim = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=args.gamma)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size= 0.01, gamma=args.gamma)
        self.checkpoint_path = args.ckpt_dir
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        
    def fit(self):
        for epoch in range(self.args.max_epochs):
            self.net.train()
            for step, inputs in enumerate(self.train_loader):
                images = inputs[1].to(self.device)
                labels = inputs[0].to(self.device).float()
                images = images.float()
                self.optim.zero_grad()
                pred = self.net(images)
                pred_max, _ = torch.max(pred, dim=1) 
                loss = self.loss_fn(pred_max, labels)
                loss.backward()
                self.optim.step()
                if (epoch+1) % self.args.print_every == 0:
                    train_acc = self.evaluate('train')
                    val_acc  = self.evaluate('val')
                    
                    print("Epoch [{}/{}] Loss: {:.3f} Train Acc: {:.3f}, Val Acc: {:.3f}".
                        format(epoch+1, self.args.max_epochs, loss.item(), train_acc, val_acc))
        
                    self.save(self.args.ckpt_name, epoch+1)
            self.scheduler.step()

    def evaluate(self, data):
        if data == 'train':
            loader = self.train_loader
        elif data == 'val':
            loader = self.val_loader

        self.net.eval()
        num_correct, num_total = 0, 0
        
        with torch.no_grad():
            for inputs in loader:
                images = inputs[1].to(self.device).float()
                labels = inputs[0].to(self.device).float()

                outputs = self.net(images)
                # print('Solver-outputs: '+ str(outputs.detach())) # => tensor([[15.4229,  3.7505], ... ,[15.4344,  3.7505]])
                # print('Solver-outputs-shape: '+str(outputs.shape)) # => torch.Size([64, 2])

                # detach() returns a new Tensor, detached from the current graph.
                # torch.max returns [value, index] along axis 1, which represents the predicted class 
                values, preds = torch.max(outputs.detach(), dim=1)
                # print('Solver-preds: '+ str(preds)) # => tensor([0, 0, 0, ... , 0]) 
                # print('Solver-preds-shape: '+str(preds.shape)) # => torch.Size([64])
                # print('Solver-preds-values: '+ str(values)) # => tensor([15.4229, 15.4268, 15.5110, ... , 15.4344])
                # print('Solver-preds-values-shape: '+str(values.shape)) # =>  torch.Size([64])

                # sum up all the correct predictions
                num_correct += (preds == labels).sum().item()
                num_total += labels.size(0)

        return num_correct / num_total

    def save(self, ckpt_name, global_step):
        save_path = os.path.join(
            self.checkpoint_path, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)
    
    def export(self):
        torch.save(self.net.state_dict(), self.args.model_dir)

    def test(self):
        test_data = Dataset(data_root=self.args.data_root, scaled=self.args.data_scaled, type='test', seed=self.args.seed)
        test_loader = DataLoader(dataset=test_data,
                                      batch_size=self.args.batch_size,
                                      num_workers=1,
                                      shuffle=False)
        self.net.eval()
        num_correct, num_total = 0, 0
        
        with torch.no_grad():
            for inputs in test_loader:
                images = inputs[1].to(self.device).float()
                labels = inputs[0].to(self.device).float()

                outputs = self.net(images)
                values, preds = torch.max(outputs.detach(), dim=1)
                num_correct += (preds == labels).sum().item()
                num_total += labels.size(0)

        return num_correct / num_total
