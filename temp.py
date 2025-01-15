import os

training_dict = {
            'epoch': [1, 2, 3], 
            'loss': [0.1, 0.2, 0.5], 
            'train_acc': [90, 91, 92], 
            'val_acc': [80, 81, 82]
        }

with open(os.path.join("training_data", "model_46.pth"[:-4] + "_training_data.csv"), "w", newline="") as f:
    # Write header
    f.write("epoch,loss,train_acc,val_acc\n")
    
    # Write data
    for i in range(len(training_dict['epoch'])):
        f.write("{},{},{},{}\n".format(training_dict['epoch'][i], training_dict['loss'][i], training_dict['train_acc'][i], training_dict['val_acc'][i]))
