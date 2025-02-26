import os
import numpy as np
import torch
import time
import sys
import linecache
from torch.utils.data import DataLoader
from net import Net, NetMini, OtherNet, OtherNetMini
from dataset import Dataset
from testDataset import TestDataset
import matplotlib.pyplot as plt


if __name__ == "__main__":
    model = sys.argv[1] # <model_name>.pth
    model_type = sys.argv[2] # one_class, two_classes
    temp = sys.argv[3] # full, single
    data_type = sys.argv[4] # train, val, test

    # Show hyperparameters of selected model
    with open(os.path.join(os.getcwd(), 'training_data', model[:-4] + "_hyper_params.csv"), "r", newline="") as f:
        # Read header
        f.readline()

        # Read hyperparameters
        f.readline()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.time()
    
    if model_type == 'one_class':
        net = OtherNet().to(device)
    elif model_type == 'one_class_mini':
        net = OtherNetMini().to(device)
    elif model_type == 'two_classes':
        net = Net().to(device)
    elif model_type == 'two_classes_mini':
        net = NetMini().to(device)
    
    net.load_state_dict(torch.load(os.path.join(os.getcwd(), 'model', model), map_location=device))
    net.eval()
    print('Time to load model: ' + str(time.time() - start_time) + ' seconds')

    # Print network architecture

    print(net)
    
    start_time = time.time()
    
    if temp == 'full':
        test_data = Dataset(data_root=os.getcwd(), scaled=True, type='test', seed=42)
        test_loader = DataLoader(dataset=test_data, batch_size=1024, num_workers=1, shuffle=False)

        net.eval()
        num_correct, num_total, TP, FP, FN = 0, 0, 0, 0, 0
        with torch.no_grad():
            for inputs in test_loader:
                if model_type == 'one_class' or model_type == 'one_class_mini':
                    images = inputs[1].to(device).float()
                    labels = inputs[0].to(device).float().view(-1, 1)
                elif model_type == 'two_classes' or model_type == 'two_classes_mini':
                    images = inputs[1].to(device)
                    labels = inputs[0].to(device).float()
                    images = images.float()

                outputs = net(images)
                
                if model_type == 'one_class' or model_type == 'one_class_mini':
                    probabilities = torch.sigmoid(outputs)
                    preds = (probabilities > 0.5).float()
                elif model_type == 'two_classes' or model_type == 'two_classes_mini':
                    values, preds = torch.max(outputs.detach(), dim=1)
                
                num_correct += (preds == labels).sum().item()
                num_total += labels.size(0)
                TP += ((preds == 1) & (labels == 1)).sum().item()
                FP += ((preds == 1) & (labels == 0)).sum().item()
                FN += ((preds == 0) & (labels == 1)).sum().item()

        acc = num_correct / num_total
        print('Test accuracy: ' + str(acc))
        f1_score = 2 * TP / (2 * TP + FP + FN)
        print('Test F1 score:', f1_score)

        print('Testing Time: ' + str(time.time() - start_time) + ' seconds')

    elif temp == 'single':      
        index = int(sys.argv[5])
        data = TestDataset(data_root=os.getcwd(), index=index, scaled=True, type='test')
        loader = DataLoader(dataset=data, batch_size=1, num_workers=1, shuffle=False)


        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook

        activations = {}
        for name, layer in net.named_children():
            layer.register_forward_hook(get_activation(name))
    
        with torch.no_grad():
            for inputs in loader:
                if model_type == 'one_class' or model_type == 'one_class_mini':
                    images = inputs[1].to(device).float()
                    labels = inputs[0].to(device).float().view(-1, 1)
                elif model_type == 'two_classes' or model_type == 'two_classes_mini':
                    images = inputs[1].to(device)
                    labels = inputs[0].to(device).float()
                    images = images.float()

                outputs = net(images)
                
                if model_type == 'one_class' or model_type == 'one_class_mini':
                    probabilities = torch.sigmoid(outputs)
                    preds = (probabilities > 0.5).float()
                elif model_type == 'two_classes' or model_type == 'two_classes_mini':
                    values, preds = torch.max(outputs.detach(), dim=1)
                
                print('Ground truth: ' + str(labels.item()) + ', Predicted label: '+ str(preds.item()))
                print('Prediction matches Ground truth: ' + str(preds.item() == labels.item()))

        # Plot
        for name, activation in activations.items():
            if name != 'fc1' and name != 'fc2':
                num_features = activation.shape[1]
                
                size = int(num_features ** 0.5)
                
                fig, axes = plt.subplots(size, size, figsize=(12, 12))   
                for i, ax in enumerate(axes.flat):
                    if i < num_features:
                        ax.imshow(activation[0, i].cpu().numpy(), cmap='viridis')
                    ax.axis('off')
                
                plt.suptitle(f'activations after layer: {name}')
                plt.show()
    else:
        print('Invalid input')
        sys.exit()


# https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
# load image
# pass it through the model
# print how each layer changes the image