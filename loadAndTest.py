import os
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from net import Net
from dataset import Dataset

if __name__ == "__main__":
    t1 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net(type='convolutional').to(device)
    net.load_state_dict(torch.load(os.path.join(os.getcwd(), 'model', 'model.pth'), map_location=device))
    test_data = Dataset(data_root=os.getcwd(), scaled=True, type='test', seed=42)
    test_loader = DataLoader(dataset=test_data, batch_size=1024, num_workers=1, shuffle=False)
    net.eval()
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