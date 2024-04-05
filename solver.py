import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from net import Net
from dataset import Dataset

class Solver():
    def __init__(self, args):
        self.args = args

        self.train_data = Dataset(data_root=args.data_root, type='train')
        self.test_data  = Dataset(data_root=args.data_root, type='val')
        
        #initialize data loader, loss function, optimizer, network
        self.train_loader = DataLoader(dataset=self.train_data,
                                       batch_size=args.batch_size,
                                       num_workers=8,
                                       shuffle=True, drop_last=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Net(type=args.type).to(self.device)
        self.loss_fn = torch.nn.MSELoss()
        self.optim = torch.optim.SGD(self.net.parameters(), lr=args.lr, weight_decay=1e-5)
        
        self.checkpoint_path = args.ckpt_dir
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        
    def fit(self):

        args = self.args

        for epoch in range(args.max_epochs):
            self.net.train()
            for step, inputs in enumerate(self.train_loader):
                images = inputs[1].to(self.device)
                """print('Solver-images: '+ str(images))
                print('Solver-images-shape: '+str(images.shape)+', ' + str(images.shape[0]))
                print('Solver-images-type: '+str(type(images)))"""
                labels = inputs[0].to(self.device).float()
                """print('Solver-labels: '+ str(labels))
                print('Solver-label-shape: '+str(labels.shape))
                print('Solver-label-type: '+str(type(labels)))"""
                images = images.float()
                pred = self.net(images)
                """print('Solver-pred: '+ str(pred))
                print('Solver-pred-shape: '+str(pred.shape))
                print('Solver-pred-type: '+str(type(pred)))"""
                pred_max, _ = torch.max(pred, dim=1)
                """print('Solver-pred-shape: '+str(pred_max.shape))
                print('Solver-pred-type: '+str(type(pred_max)))"""
                loss = self.loss_fn(pred_max, labels)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                if (epoch+1) % args.print_every == 0:
                    train_acc = self.evaluate(self.train_data)
                    test_acc  = self.evaluate(self.test_data)
                    
                    print("Epoch [{}/{}] Loss: {:.3f} Train Acc: {:.3f}, Test Acc: {:.3f}".
                        format(epoch+1, args.max_epochs, loss.item(), train_acc, test_acc))

                    self.save(args.ckpt_name, epoch+1)

    def evaluate(self, data):
        args = self.args
        
        loader = DataLoader(dataset=data,
                            batch_size=args.batch_size,
                            num_workers=1,
                            shuffle=False)

        self.net.eval()
        num_correct, num_total = 0, 0
        
        with torch.no_grad():
            for inputs in loader:
                images = inputs[1].to(self.device)
                labels = inputs[0].to(self.device)

                outputs = self.net(images)
                
                # detach() returns a new Tensor, detached from the current graph.
                # torch.max returns [value, index] along axis 1, which represents the predicted class 
                _, preds = torch.max(outputs.detach(), 1)

                # sum up all the correct predictions
                num_correct += (preds == labels).sum().item()
                num_total += labels.size(0)

        return num_correct / num_total
        
        
    def save(self, ckpt_name, global_step):
        save_path = os.path.join(
            self.checkpoint_path, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)
