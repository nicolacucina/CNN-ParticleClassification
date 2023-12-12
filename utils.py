import os
import uproot as up
import numpy as np
import matplotlib.pyplot as plt
import linecache
from math import floor, ceil
import random

def convert_root_to_csv(data_root, energy_flag=False):
    dataset_flag = False
    
    if not os.path.exists(os.path.join(data_root, 'data')):
        os.makedirs(os.path.join(data_root, 'data'))
        dataset_flag = True
        if not os.path.exists(os.path.join(data_root, 'data', 'dataset_csv')):
            dataset_csv = open(data_root+'/data/dataset_csv', "a")
        if energy_flag:
            dataset_csv = open(data_root+'/data/dataset_energy_csv', "a")

    if dataset_flag:
        print('Creating dataset_csv')
        for filename in os.listdir('enebins'):
            with open(os.path.join('enebins', filename)) as f:
                if filename.endswith(".root"):
                    print('processing '+filename)
                    if filename.startswith("e_"):
                        e_file = up.open(os.path.join(os.getcwd(), 'enebins', filename))
                        # This is the complete file
                        # shower = e_file["showers"].array(library="np") # => dict
                        
                        if energy_flag:
                            energy = e_file["showers"]["energy"].array(library="np")
                        
                        # We are using only the deposited energy
                        dep = e_file["showers"]["dep"].array(library="np")
                        for i in range(len(dep)):
                            temp = '0,' # 0 is the label for electron
                            
                            if energy_flag:
                                temp = temp + str(energy[i]) + ','

                            for j in range(len(dep[i])):
                                temp = temp + str(dep[i][j]) + ','
                            temp = temp[:-1]
                            dataset_csv.write(temp + '\n')

                    if filename.startswith("p_"):
                        p_file = up.open(os.path.join(os.getcwd(), 'enebins', filename))
                        
                        if energy_flag:
                            energy = p_file["showers"]["energy"].array(library="np")
                        
                        dep = p_file["showers"]["dep"].array(library="np")
                        for i in range(len(dep)):
                            temp = '1,' # 1 is the label for proton
                            
                            if energy_flag:
                                temp = temp + str(energy[i]) + ','

                            for j in range(len(dep[i])):
                                temp = temp + str(dep[i][j]) + ','
                            temp = temp[:-1]
                            dataset_csv.write(temp + '\n')                        
        dataset_csv.close()
    else:
        print('dataset_csv already exists')                

def splitTrainTest(data_root, energy_flag=False, size=10627):
    if energy_flag:
        #dataset_name = 'dataset_energy_csv'
        train_name = 'dataset_energy_csv_train'
        test_name = 'dataset_energy_csv_test'
    else: 
        #dataset_name = 'dataset_csv'
        train_name = 'dataset_csv_train'
        test_name = 'dataset_csv_test'

    if (os.path.exists(os.path.join(data_root, 'data', train_name))) and (os.path.exists(os.path.join(data_root, 'data', test_name))):
        print(train_name + ' and '+ test_name +' already exist')
    else:
        #dataset_csv = open(data_root+'/data/'+dataset_name, "r")
        dataset_csv_train = open(data_root+'/data/'+train_name, "a")
        dataset_csv_test = open(data_root+'/data/'+test_name, "a")

        for i in range(size):
            line = linecache.getline(data_root+'/data/'+dataset_name, i)
            if i%8 == 0:
                dataset_csv_test.write(line)
            else:
                dataset_csv_train.write(line)
        dataset_csv_train.close()
        dataset_csv_test.close()
        #dataset_csv.close()

def plotExample(data_root, amount, energy_flag=False, size=10627):
    if amount%2 != 0:
        print('amount must be even')
        return
    else:
        if energy_flag:
            dataset_name = 'dataset_energy_csv'
        else: 
            dataset_name = 'dataset_csv'
        
        dataset_csv = open(data_root+'/data/'+dataset_name, "r")
        fig = plt.figure(figsize=(8, 8))
        rows = floor(amount/2) if amount%2 == 0 else ceil(amount/2)
        columns = floor(amount/rows)
        for i in range(amount):
            index = random.randint(0, size)
            line = linecache.getline(data_root+'/data/'+dataset_name, index)
            line = line.split(',')
            print('line dimension: '+ str(len(line)))
            target = line[0]
            print('target: electron' if target == '0' else 'target: proton')
            dep = line[1:]
            dep = np.array(dep, dtype='float64').reshape(20,20)
            #print('Deposit:')
            #print(dep)
            print('Shape of deposit: '+str(dep.shape))
            print('data type: ' + str(dep.dtype))

            fig.add_subplot(rows, columns, i+1)
            plt.xlabel('r coordinate')
            plt.ylabel('t coordinate')
            plt.imshow(dep, cmap='hot', interpolation='nearest')
            plt.title('electron' if target == '0' else 'proton')
        
        plt.subplots_adjust(hspace=0.5)
        plt.show()

if __name__ == "__main__":
    convert_root_to_csv(os.getcwd())
    splitTrainTest(os.getcwd())
    plotExample(os.getcwd(), amount=6)
