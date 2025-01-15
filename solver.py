import os
import numpy as np
import torch
import matplotlib.pyplot as plt
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
        
        # Dict to log training metrics
        self.training_dict = {
            'epoch': [], 
            'loss': [], 
            'train_acc': [], 
            'val_acc': []
            'train_f1': [],
            'val_f1': []
        }

        # Initialize data loader, loss function, optimizer, network
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
        self.net = Net().to(self.device)
        # self.net = Net1().to(self.device)
        
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

                    train_acc, train_f1 = self.evaluate('train')
                    val_acc, val_f1  = self.evaluate('val')
                    
                    self.training_dict['epoch'].append(epoch+1)
                    self.training_dict['loss'].append(loss.item())
                    self.training_dict['train_acc'].append(train_acc)
                    self.training_dict['val_acc'].append(val_acc)
                    self.training_dict['train_f1'].append(train_f1)
                    self.training_dict['val_f1'].append(val_f1)

                    print("Epoch [{}/{}], Loss: {:.3f}, Train Acc: {:.3f}, Val Acc: {:.3f}, Train F1: {:.3f}, Val F1: {:.3f}".
                        format(epoch+1, self.args.max_epochs, loss.item(), train_acc, val_acc, train_f1, val_f1))

                    self.save(self.args.ckpt_name, epoch+1)
            self.scheduler.step()
        
        # Save the dictionary after training
        with open(os.path.join("training_data", self.args.model_dir[:-4] + "_training_data.csv"), "w", newline="") as f:
            # Write header
            f.write("epoch,loss,train_acc,val_acc\n")
            
            # Write data
            for i in range(len(training_dict['epoch'])):
                f.write("{},{},{},{}\n".format(training_dict['epoch'][i], training_dict['loss'][i], training_dict['train_acc'][i], training_dict['val_acc'][i]))

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

                # Debugging prints
                # print('Solver-outputs: '+ str(outputs.detach())) # => tensor([[15.4229,  3.7505], ... ,[15.4344,  3.7505]])
                # print('Solver-outputs-shape: '+str(outputs.shape)) # => torch.Size([64, 2])

                # detach() returns a new Tensor, detached from the current graph.

                # torch.max returns [value, index] along axis 1, where index represents the predicted class 
                values, preds = torch.max(outputs.detach(), dim=1)
                
                # Debugging prints
                # print('Solver-preds: '+ str(preds)) # => tensor([0, 0, 0, ... , 0]) 
                # print('Solver-preds-shape: '+str(preds.shape)) # => torch.Size([64])
                # print('Solver-preds-values: '+ str(values)) # => tensor([15.4229, 15.4268, 15.5110, ... , 15.4344])
                # print('Solver-preds-values-shape: '+str(values.shape)) # =>  torch.Size([64])

                # Sum up all the correct predictions
                num_correct += (preds == labels).sum().item()
                num_total += labels.size(0)
                
                acc = num_correct / num_total

                # Compute f1_score
                TP = ((preds == 1) & (labels == 1)).sum().item()
                FP = ((preds == 1) & (labels == 0)).sum().item()
                FN = ((preds == 0) & (labels == 1)).sum().item()

                f1_score = 2 * TP / (2 * TP + FP + FN)

        return acc, f1_score
        # return acc, f1_score(preds, labels)

    def save(self, ckpt_name, global_step):
        save_path = os.path.join(
            self.checkpoint_path, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)
    
    def print_plot(self):
        plt.plot(self.training_dict['epoch'], self.training_dict['loss'], label='Loss')
        plt.plot(self.training_dict['epoch'], self.training_dict['train_acc'], label='Train Accuracy')
        plt.plot(self.training_dict['epoch'], self.training_dict['val_acc'], label='Val. Accuracy')
        plt.plot(self.training_dict['epoch'], self.training_dict['train_f1'], label='Train F1_Score')
        plt.plot(self.training_dict['epoch'], self.training_dict['val_f1'], label='Val. F1_Score')
        plt.xlabel('Epoch')
        plt.ylabel('Metrics')
        plt.legend()
        plt.show()

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

            print('Test accuracy: ', (num_correct / num_total))