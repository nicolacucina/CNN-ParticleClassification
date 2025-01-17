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
        self.val_data = Dataset(data_root=args.data_root, scaled=args.data_scaled, type='val', seed=args.seed)
        
        # Dict to log training metrics
        self.training_dict = {
            'epoch': [], 
            'loss': [], 
            'train_acc': [], 
            'val_acc': [],
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
        
        if self.args.type == 'one_class':
            print('Net1 choosen')
            self.net = Net1().to(self.device)
            self.loss_fn = torch.nn.BCEWithLogitsLoss() # combines a Sigmoid layer and the BCELoss in one single class
        elif self.args.type == 'two_classes':
            print('Net choosen')
            self.net = Net().to(self.device)
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            exit()
        # https://stackoverflow.com/questions/53628622/loss-function-its-inputs-for-binary-classification-pytorch
        
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
                # images = inputs[1].to(self.device)
                # labels = inputs[0].to(self.device).float().view(-1, 1)
                # images = images.float()
                if self.args.type == 'one_class':   
                    images = inputs[1].to(self.device)
                    labels = inputs[0].to(self.device).float().view(-1, 1)
                    images = images.float()
                elif self.args.type == 'two_classes':
                    continue
                self.optim.zero_grad()
                pred = self.net(images)
                # pred_max, _ = torch.max(pred, dim=1) # mi sa che questo va tolto perchè non lo fa mai il professore
                # intanto capire se prendo il valore oppure l'indice 
                # ma devo lasciare due neuroni e togliere max? oppure lo lascio?
                # in teoria funziona in entrambi i modi
                # https://stackoverflow.com/questions/66906884/how-is-pytorchs-class-bcewithlogitsloss-exactly-implemented
                # loss = self.loss_fn(pred_max, labels)
                
                loss = self.loss_fn(pred, labels)
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

                # If checkpointing is needed
                # self.save(self.args.ckpt_name, epoch+1)
            self.scheduler.step()

    def evaluate(self, data):
        if data == 'train':
            loader = self.train_loader
        elif data == 'val':
            loader = self.val_loader

        self.net.eval()
        num_correct, num_total, TP, FP, FN = 0, 0, 0, 0, 0
        
        with torch.no_grad():
            for inputs in loader:
                # images = inputs[1].to(self.device).float()
                # labels = inputs[0].to(self.device).float()
                if self.args.type == 'one_class':
                    images = inputs[1].to(self.device).float()
                    labels = inputs[0].to(self.device).float().view(-1, 1)

                outputs = self.net(images)

                # Debugging prints
                # print('Solver-outputs: '+ str(outputs.detach())) 
                # tensor([[15.4229,  3.7505], ... ,[15.4344,  3.7505]])
                
                # print('Solver-outputs-shape: '+str(outputs.shape))
                # torch.Size([64, 2])

                # detach() returns a new Tensor, detached from the current graph.

                # torch.max returns [value, index] along axis 1, where index represents the predicted class 
                if self.args.type == 'one_class':
                    probabilities = torch.sigmoid(outputs)
                    preds = (probabilities > 0.5).float()

                # Debugging prints
                # print('Solver-preds: '+ str(preds))
                # tensor([0, 0, 0, ... , 0]) 
                
                # print('Solver-preds-shape: '+str(preds.shape))
                # torch.Size([64])
                
                # print('Solver-preds-values: '+ str(values))
                # tensor([15.4229, 15.4268, 15.5110, ... , 15.4344])
                
                # print('Solver-preds-values-shape: '+str(values.shape))
                # torch.Size([64])

                # Sum up all the correct predictions
                num_correct += (preds == labels).sum().item()
                num_total += labels.size(0)
                
                        
                # Compute f1_score
                TP += ((preds == 1) & (labels == 1)).sum().item()
                FP += ((preds == 1) & (labels == 0)).sum().item()
                FN += ((preds == 0) & (labels == 1)).sum().item()

        acc = num_correct / num_total
        f1_score = 2 * TP / (2 * TP + FP + FN)

        return acc, f1_score

    # If checkpointing is needed
    def save(self, ckpt_name, global_step):
        save_path = os.path.join(
            self.checkpoint_path, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)
    
    def print_plot(self):
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        # Plot the Loss
        axs[0].plot(self.training_dict['epoch'], self.training_dict['loss'], label='Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        # Plot the metrics
        axs[1].plot(self.training_dict['epoch'], self.training_dict['train_acc'], label='Train Accuracy')
        axs[1].plot(self.training_dict['epoch'], self.training_dict['val_acc'], label='Validation Accuracy')
        axs[1].plot(self.training_dict['epoch'], self.training_dict['train_f1'], label='Train F1 Score')
        axs[1].plot(self.training_dict['epoch'], self.training_dict['val_f1'], label='Validation F1 Score')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Metrics')
        axs[1].legend()

        plt.show()

    def export(self):
        # Save the model
        torch.save(self.net.state_dict(), os.path.join(self.args.model_dir, self.args.model_name))
        
        # Save the dictionary
        with open(os.path.join(self.args.training_data_dir, self.args.model_name[:-4] + "_training_data.csv"), "w", newline="") as f:
            f.write("epoch,loss,train_acc,val_acc,train_f1,val_f1\n")

            for i in range(len(self.training_dict['epoch'])):
                f.write("{},{},{},{},{},{}\n".format(self.training_dict['epoch'][i],
                                                self.training_dict['loss'][i], 
                                                self.training_dict['train_acc'][i], 
                                                self.training_dict['val_acc'][i],
                                                self.training_dict['train_f1'][i], 
                                                self.training_dict['val_f1'][i]))
        
        # Save the hyperparams
        with open(os.path.join(self.args.training_data_dir, self.args.model_name[:-4] + "_hyper_params.csv"), "w", newline="") as f:
            f.write("batch_size,max_epochs,lr,decay,gamma\n")

            f.write("{},{},{},{},{}".format(self.args.batch_size, 
                                            self.args.max_epochs, 
                                            self.args.lr, 
                                            self.args.decay, 
                                            self.args.gamma))

    def test(self):
        test_data = Dataset(data_root=self.args.data_root, scaled=self.args.data_scaled, type='test', seed=self.args.seed)
        test_loader = DataLoader(dataset=test_data,
                                      batch_size=self.args.batch_size,
                                      num_workers=1,
                                      shuffle=False)
        self.net.eval()
        num_correct, num_total, TP, FP, FN = 0, 0, 0, 0, 0
        with torch.no_grad():
            for inputs in test_loader:
                if self.args.type == 'one_class':
                    images = inputs[1].to(self.device).float()
                    labels = inputs[0].to(self.device).float().view(-1, 1)

                outputs = self.net(images)
                
                if self.args.type == 'one_class':
                    probabilities = torch.sigmoid(outputs)
                    preds = (probabilities > 0.5).float()

                num_correct += (preds == labels).sum().item()
                num_total += labels.size(0)

                TP += ((preds == 1) & (labels == 1)).sum().item()
                FP += ((preds == 1) & (labels == 0)).sum().item()
                FN += ((preds == 0) & (labels == 1)).sum().item()

        acc = (num_correct / num_total)
        print('Test accuracy: ', acc)
        f1_score = 2 * TP / (2 * TP + FP + FN)
        print('Test F1 Score: ', f1_score)