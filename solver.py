import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from net import Net
from dataset import Dataset

class Solver():
    def __init__(self, args):
        self.args = args
        self.train_data = Dataset(data_root=args.data_root, scaled=args.data_scaled, type='train', seed=args.seed)
        self.test_data  = Dataset(data_root=args.data_root, scaled=args.data_scaled, type='val', seed=args.seed)
        
        #initialize data loader, loss function, optimizer, network
        self.train_loader = DataLoader(dataset=self.train_data,
                                       batch_size=args.batch_size,
                                       num_workers=1,
                                       shuffle=True, drop_last=True)

        self.test_loader = DataLoader(dataset=self.test_data,
                                      batch_size=args.batch_size,
                                      num_workers=1,
                                      shuffle=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Net(type=args.type).to(self.device)
        # self.loss_fn = torch.nn.MSELoss()
        # self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_fn = torch.nn.BCEWithLogitsLoss() # combines a Sigmoid layer and the BCELoss in one single class

        self.optim = torch.optim.SGD(self.net.parameters(), lr=args.lr, weight_decay=1e-5)     
        #self.optim = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=1e-5)
        
        self.checkpoint_path = args.ckpt_dir
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        
    def fit(self):

        # controlla parametri dei layer se cambiano tra epoche
        for epoch in range(self.args.max_epochs):
            self.net.train()
            for step, inputs in enumerate(self.train_loader):
                # print('Solver-inputs: '+ str(inputs[0]))

                images = inputs[1].to(self.device)
                # print('Solver-images: '+ str(images))
                # print('Solver-images-shape: '+str(images.shape)+', ' + str(images.shape[0]))
                # print('Solver-images-type: '+str(type(images)))

                labels = inputs[0].to(self.device).float()
                # print('Solver-labels: '+ str(labels))
                # print('Solver-label-shape: '+str(labels.shape))
                # print('Solver-label-type: '+str(type(labels)))

                images = images.float()
                pred = self.net(images)
                # print('Solver-pred: '+ str(pred)) # => tensor([[ 0.0341, -0.1604],, ... , grad_fn=<MaxBackward0>)
                # print('Solver-pred-shape: '+str(pred.shape))
                # print('Solver-pred-type: '+str(type(pred)))

                pred_max, _ = torch.max(pred, dim=1) 
                # print('Solver-pred-max: '+ str(pred_max)) # => tensor([ 0.0341, ... ,grad_fn=<MaxBackward0>)               
                # print('Solver-pred-max-shape: '+str(pred_max.shape))
                # print('Solver-pred-max-type: '+str(type(pred_max)))
                loss = self.loss_fn(pred_max, labels)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                if (epoch+1) % self.args.print_every == 0:
                    train_acc = self.evaluate('train')
                    test_acc  = self.evaluate('test')
                    
                    print("Epoch [{}/{}] Loss: {:.3f} Train Acc: {:.3f}, Test Acc: {:.3f}".
                        format(epoch+1, self.args.max_epochs, loss.item(), train_acc, test_acc))
                               
                    # Epoch [1/3] Loss: 148.178 Train Acc: 0.525, Test Acc: 0.524
                    # Epoch [1/3] Loss: 103.751 Train Acc: 0.525, Test Acc: 0.524
                    # Epoch [1/3] Loss: 111.480 Train Acc: 0.524, Test Acc: 0.524
                    # Epoch [1/3] Loss: 131.251 Train Acc: 0.525, Test Acc: 0.524
                    # Epoch [1/3] Loss: 120.198 Train Acc: 0.525, Test Acc: 0.524
        
                    self.save(self.args.ckpt_name, epoch+1)

    def evaluate(self, data):
        if data == 'train':
            loader = self.train_loader
        elif data == 'test':
            loader = self.test_loader

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
        torch.save(self.net.state_dict(), args.model_dir)

    def test(self):
        test_data = Dataset(data_root=args.data_root, type='test', seed=args.seed)
        test_loader = DataLoader(dataset=test_data,
                                      batch_size=args.batch_size,
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
