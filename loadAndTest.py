# python3 loadAndTest.py model30.pth full train
import os
import numpy as np
import torch
import time
import sys
import linecache
from torch.utils.data import DataLoader
from net import Net
from dataset import Dataset
from testDataset import TestDataset

if __name__ == "__main__":
    model = sys.argv[1]
    temp = sys.argv[2]
    type = sys.argv[3]
    t1 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)
    net.load_state_dict(torch.load(os.path.join(os.getcwd(), 'model', model), map_location=device))
    net.eval()
    if temp == 'full':
        test_data = Dataset(data_root=os.getcwd(), scaled=True, type=type, seed=42)
        test_loader = DataLoader(dataset=test_data, batch_size=1024, num_workers=1, shuffle=False)
        
        num_correct, num_total = 0, 0

        with torch.no_grad():
            for inputs in test_loader:
                images = inputs[1].to(device).float()
                labels = inputs[0].to(device).float()

                outputs = net(images)
                values, preds = torch.max(outputs.detach(), dim=1)
                num_correct += (preds == labels).sum().item()
                num_total += labels.size(0)
        
        acc = num_correct / num_total
        print('Test accuracy: ' + str(acc))
        t2 = time.time()
        print('Testing Time: ' + str(t2 - t1) + ' seconds')
    elif temp == 'single':      
        index = int(sys.argv[4])
        data = TestDataset(data_root=os.getcwd(), index=index, scaled=True, type=type)
        loader = DataLoader(dataset=data, batch_size=1, num_workers=1, shuffle=False)
        with torch.no_grad():
            for inputs in loader:
                # print(inputs)
                images = inputs[1].to(device).float()
                labels = inputs[0].to(device).float()
                pred = net(images)
                values, preds = torch.max(pred.detach(), dim=1)
                print('Ground truth: ' + str(labels.item()) + ', Predicted label: '+ str(preds.item()))
                print('Prediction matches Ground truth: ' + str(preds.item() == labels.item()))
    else:
        print('Invalid input')
        sys.exit()
