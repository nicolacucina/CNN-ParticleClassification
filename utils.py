import os
import uproot as up
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import linecache
from math import floor, ceil
import random

def convert_root_to_csv(data_root, energy_flag=False):
    dataset_flag = False

    # Create the data folder if it does not exist
    if not os.path.exists(os.path.join(data_root, 'data')):
        os.makedirs(os.path.join(data_root, 'data'))
        dataset_flag = True
    
    # Create the dataset.csv file if it does not exist
    if not os.path.exists(os.path.join(data_root, 'data', 'dataset.csv')):
        dataset_csv = open(os.path.join(data_root, 'data', 'dataset.csv'), "a")
        dataset_flag = True

    if energy_flag:
        if not os.path.exists(os.path.join(data_root, 'data', 'dataset_energy.csv')):
            dataset_csv = open(os.path.join(data_root, 'data', 'dataset_energy.csv'), "a")
            dataset_flag = True

    if dataset_flag:
        print('Creating dataset.csv')
        index = 0
        for filename in os.listdir('enebins'):
            with open(os.path.join('enebins', filename)) as f:
                if filename.endswith(".root"):
                    print('processing '+filename)
                    if filename.startswith("e_"):
                        e_file = up.open(os.path.join(os.getcwd(), 'enebins', filename))
                        
                        #print(e_file.keys()) # => ['showers;1']
                        #print(e_file["showers"].keys()) # => ['id', 'dep', 'E0']
                        
                        if energy_flag:
                            energy = e_file["showers"]["energy"].array(library="np")
                        
                        # We are using only the deposited energy
                        dep = e_file["showers"]["dep"].array(library="np")
                        #print(dep)
                        print(dep.shape) 
                        
                        for i in range(len(dep)):
                            
                            #Check is the shape is correct 
                            #print(dep[i].shape) #=> (400,)
                            if dep[i].shape[0] == 400:
                                temp = '0,' # 0 is the label for electron
            
                                if energy_flag:
                                    temp = temp + str(energy[i]) + ','

                                invalid_flag = False
                                for j in range(len(dep[i])):
                                    if dep[i][j] >= 0:
                                        temp = temp + str(dep[i][j]) + ','
                                    else:
                                        invalid_flag = True
                                        print('Negative value found at event ' + str(index) + ' in file ' + filename)

                                temp = temp[:-1]

                                #Check if after removing the last comma the shape is correct
                                if len(temp.split(',')) == 401:
                                    dataset_csv.write(temp + '\n')    
                            else:
                                print('Wrong shape found')

                            if not invalid_flag:
                                index += 1

                    if filename.startswith("p_"):
                        p_file = up.open(os.path.join(os.getcwd(), 'enebins', filename))
                        
                        #print(p_file.keys()) # => ['showers;1']
                        #print(p_file["showers"].keys()) # => ['id', 'dep', 'E0']

                        if energy_flag:
                            energy = p_file["showers"]["energy"].array(library="np")
                        
                        dep = p_file["showers"]["dep"].array(library="np")
                        #print(dep)
                        print(dep.shape) 
                        
                        for i in range(len(dep)):
                            
                            #Check is the shape is correct 
                            #print(dep[i].shape) #=> (400,)
                            if dep[i].shape[0] == 400:
                                temp = '1,' # 1 is the label for proton
                                
                                if energy_flag:
                                    temp = temp + str(energy[i]) + ','

                                invalid_flag = False
                                for j in range(len(dep[i])):
                                    if dep[i][j] >= 0:
                                        temp = temp + str(dep[i][j]) + ','
                                    else:
                                        invalid_flag = True
                                        print('Negative value found at event ' + str(index) + ' in file ' + filename)
                                    
                                temp = temp[:-1]

                                #Check if after removing the last comma the shape is correct
                                if len(temp.split(',')) == 401:
                                    dataset_csv.write(temp + '\n')
                            else:
                                print('Wrong shape found')  

                            if not invalid_flag:
                                index += 1                    
        dataset_csv.close()
        print('dataset.csv created')
        print('dataset.csv size: ' + str(index))

    else:
        print('dataset.csv already exists')

def splitTrainTest(data_root, scaled=False, energy_flag=False, size=10626, mod=8):
    # splittare meglio -> ogni tanto ottengo f1 score nullo
    if scaled:
        if energy_flag:
            #dataset_name = 'dataset_energy_scaled.csv'
            train_name = 'dataset_energy_scaled_train.csv'
            val_name = 'dataset_energy_scaled_val.csv'
            test_name = 'dataset_energy_scaled_test.csv'
        else: 
            dataset_name = 'dataset_scaled.csv'
            train_name = 'dataset_scaled_train.csv'
            val_name = 'dataset_scaled_val.csv'
            test_name = 'dataset_scaled_test.csv'
    else:
        if energy_flag:
            #dataset_name = 'dataset_energy.csv'
            train_name = 'dataset_energy_train.csv'
            val_name = 'dataset_energy_val.csv'
            test_name = 'dataset_energy_test.csv'
        else: 
            dataset_name = 'dataset.csv'
            train_name = 'dataset_train.csv'
            val_name = 'dataset_val.csv'
            test_name = 'dataset_test.csv'

    if (os.path.exists(os.path.join(data_root, 'data', train_name))) and (os.path.exists(os.path.join(data_root, 'data', test_name))) and (os.path.exists(os.path.join(data_root, 'data', val_name))):
        print(train_name + ', '+ val_name + ' and '+ test_name +' already exist')
    else:
        dataset_csv = open(data_root+'/data/'+dataset_name, "r")
        dataset_csv_train = open(data_root+'/data/'+train_name, "a")
        dataset_csv_val = open(data_root+'/data/'+val_name, "a")
        dataset_csv_test = open(data_root+'/data/'+test_name, "a")

        j = 0
        for i in range(size):
            line = linecache.getline(data_root+'/data/'+dataset_name, i)
            if i%mod == 0:
                dataset_csv_test.write(line)
            else:
                if j == mod:
                    j = 0
                    dataset_csv_val.write(line)
                else:
                    j += 1
                    dataset_csv_train.write(line)
        
        dataset_csv_train.close()
        dataset_csv_val.close()
        dataset_csv_test.close()
        dataset_csv.close()

def countClasses(dataset_path, scaled=True, energy_flag=False, size=10626):
    dataset_csv = open(dataset_path, "r")
    count_electron = 0
    count_proton = 0
    for i in range(size):
        line = linecache.getline(dataset_path, i)
        line = line.split(',')
        if line[0] == '0':
            count_electron += 1
        else:
            count_proton += 1
    print('Electrons: ' + str(count_electron) + ', Protons: ' + str(count_proton))

def findMaxMin(data_root, energy_flag=False, size=10626):
    if energy_flag:
        dataset_name = 'dataset_energy_min_max.txt'
    else: 
        dataset_name = 'dataset_min_max.txt'

    if(os.path.exists(data_root + '/data/' +  dataset_name) == False):
        print('Max and Min do not exist')
        dataset_min_max = open(os.path.join(data_root, 'data', dataset_name), "a")
        dataset_csv = open(data_root + '/data/' +  'dataset.csv', "r")
    
        max_value = -1 # Energy is always positive
        min_value = 1000000 # Big enough number        

        with dataset_csv as f:
            for i in range(size):
                line = f.readline()
                line = line.split(',')
                dep = line[1:]
                dep = np.array(dep, dtype='float64')
                dep_nonzero = dep[dep != 0.0]
                if dep_nonzero.max() > max_value:
                    print('Max value found at event ' + str(i))
                    max_value = dep_nonzero.max()
                    
                if (dep_nonzero.min() < min_value):
                    print('Min value found at event ' + str(i))
                    min_value = dep_nonzero.min()
        dataset_min_max.write('Max value: '+str(max_value) + ', Min value: '+str(min_value))
        dataset_min_max.close()
        dataset_csv.close()
    else:
        print('Max and Min already exist')
        dataset_min_max = open(data_root + '/data/' +  dataset_name, "r")
        line = dataset_min_max.readline()
        line = line.split(',')
        max_value = float(line[0].split(':')[1])
        min_value = float(line[1].split(':')[1])   
        dataset_min_max.close()
    print('Max value: '+str(max_value) + ', Min value: '+str(min_value))

def scaleDataset(data_root, scalefactor=1000000, energy_flag=False, size=10626):
    if energy_flag:
        dataset_name = 'dataset_energy_scaled.csv'
    else: 
        dataset_name = 'dataset_scaled.csv'

    if(os.path.exists(data_root + '/data/' +  dataset_name) == False):
        print('Scaled dataset does not exist')
        dataset_csv_scaled = open(os.path.join(data_root, 'data', dataset_name), "a")
        dataset_csv = open(data_root + '/data/' +  'dataset.csv', "r")
    
        with dataset_csv as f:
            for i in range(size):
                line = f.readline()
                line = line.split(',')
                target = line[0]
                dep = line[1:]
                dep = np.array(dep, dtype='float64')

                dep_scaled = dep * scalefactor # <-- Scaling function
                # The scaling can be done in many ways, 
                # for now i want to keep the range between max and min values the same, 
                # even if this means having max values in the millions 
                # while the min values are in the unit range
                # normalizzare meglio?
                temp = target + ','
                for i in range(len(dep_scaled)):
                    temp = temp + str(dep_scaled[i]) + ','
                temp = temp[:-1]
                dataset_csv_scaled.write(temp + '\n')
        dataset_csv_scaled.close()
        dataset_csv.close()
    else:
        print('Scaled dataset already exists')

def plotExample(data_root, amount, scaled=False, energy_flag=False, size=10626):
    if amount%2 != 0:
        print('Amount must be even')
        return
    else:
        if scaled:
            if energy_flag:
                dataset_name = 'dataset_energy_scaled.csv'
            else: 
                dataset_name = 'dataset_scaled.csv'
        else:
            if energy_flag:
                dataset_name = 'dataset_energy.csv'
            else: 
                dataset_name = 'dataset.csv'
        
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
            print('Deposit:')
            print(str(dep)[:100], '...', str(dep)[-100:]) 
            print('Max value: '+str(np.max(dep)) + ', Min value: '+str(np.min(dep[dep != 0.0])))
            print('Shape of deposit: '+str(dep.shape))
            print('data type: ' + str(dep.dtype))

            fig.add_subplot(rows, columns, i+1)
            plt.xlabel('r coordinate')
            plt.ylabel('t coordinate')
            plt.imshow(dep, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title('electron' if target == '0' else 'proton')
        
        plt.subplots_adjust(hspace=0.5)
        plt.show()

def plotHistogram(data_root, scaled=False, energy_flag=False, size=10626):
    if scaled:
        if energy_flag:
            dataset_name = 'dataset_energy_scaled.csv'
        else: 
            dataset_name = 'dataset_scaled.csv'
    else:
        if energy_flag:
            dataset_name = 'dataset_energy.csv'
        else: 
            dataset_name = 'dataset.csv'
    
    data = pd.read_csv(os.path.join(data_root, 'data', dataset_name), header=None)
    labels = data.iloc[:, 0]
    labels.info()
    non_zero = data.iloc[:, 1:].values.flatten()
    non_zero = non_zero[non_zero != 0.0] 
    
    # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plt.hist(non_zero, bins=1000, color='blue')
    plt.title('Deposit energy distribution')
    plt.xlabel('Deposit energy')
    plt.ylabel('Count')

    # ax[1].bar([0, 1], [len(labels[labels == 0]), len(labels[labels == 1])], color='blue')
    # ax[1].set_title('Labels distribution')
    # ax[1].set_xlabel('Labels')
    # ax[1].set_ylabel('Count')

    plt.show()

if __name__ == "__main__":
    convert_root_to_csv(os.getcwd())
    findMaxMin(os.getcwd())
    splitTrainTest(os.getcwd(), scaled=True)
    countClasses(os.path.join(os.getcwd(), 'data', 'dataset.csv')) # 10626 -> Electrons: 5572, Protons: 5054
    countClasses(os.path.join(os.getcwd(), 'data', 'dataset_train.csv'), size=8264) #8264 -> Electrons: 4335, Protons: 3929
    countClasses(os.path.join(os.getcwd(), 'data', 'dataset_val.csv'), size=1033) #1033 -> Electrons: 541, Protons: 492
    countClasses(os.path.join(os.getcwd(), 'data', 'dataset_test.csv'), size=1328) #1328 -> Electrons: 696, Protons: 632
    scaleDataset(os.getcwd())
    plotHistogram(os.getcwd())
    plotExample(os.getcwd(), amount=6)