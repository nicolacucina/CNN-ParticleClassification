import os
import uproot as up
import numpy as np
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
    
    # Create the dataset_csv file if it does not exist
    if not os.path.exists(os.path.join(data_root, 'data', 'dataset_csv')):
        dataset_csv = open(os.path.join(data_root, 'data', 'dataset_csv'), "a")
        dataset_flag = True

    if energy_flag:
        if not os.path.exists(os.path.join(data_root, 'data', 'dataset_energy_csv')):
            dataset_csv = open(os.path.join(data_root, 'data', 'dataset_energy_csv'), "a")
            dataset_flag = True

    if dataset_flag:
        print('Creating dataset_csv')
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
        print('dataset_csv created')
        print('dataset_csv size: ' + str(index))

    else:
        print('dataset_csv already exists')

def splitTrainTest(data_root, scaled=False, energy_flag=False, size=10626, mod=8):
    if scaled:
        if energy_flag:
            #dataset_name = 'dataset_energy_scaled_csv'
            train_name = 'dataset_energy_csv_scaled_train'
            val_name = 'dataset_energy_csv_scaled_val'
            test_name = 'dataset_energy_csv_scaled_test'
        else: 
            dataset_name = 'dataset_csv_scaled'
            train_name = 'dataset_csv_scaled_train'
            val_name = 'dataset_csv_scaled_val'
            test_name = 'dataset_csv_scaled_test'
    else:
        if energy_flag:
            #dataset_name = 'dataset_energy_csv'
            train_name = 'dataset_energy_csv_train'
            val_name = 'dataset_energy_csv_val'
            test_name = 'dataset_energy_csv_test'
        else: 
            dataset_name = 'dataset_csv'
            train_name = 'dataset_csv_train'
            val_name = 'dataset_csv_val'
            test_name = 'dataset_csv_test'

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

def findMaxMin(data_root, energy_flag=False, size=10626):
    if energy_flag:
        dataset_name = 'dataset_energy_csv_min_max'
    else: 
        dataset_name = 'dataset_csv_min_max'

    if(os.path.exists(data_root + '/data/' +  dataset_name) == False):
        print('Max and Min do not exist')
        dataset_min_max = open(os.path.join(data_root, 'data', dataset_name), "a")
        dataset_csv = open(data_root + '/data/' +  'dataset_csv', "r")
    
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
        dataset_name = 'dataset_energy_csv_scaled'
    else: 
        dataset_name = 'dataset_csv_scaled'

    if(os.path.exists(data_root + '/data/' +  dataset_name) == False):
        print('Scaled dataset does not exist')
        dataset_csv_scaled = open(os.path.join(data_root, 'data', dataset_name), "a")
        dataset_csv = open(data_root + '/data/' +  'dataset_csv', "r")
    
        with dataset_csv as f:
            for i in range(size):
                line = f.readline()
                line = line.split(',')
                target = line[0]
                dep = line[1:]
                dep = np.array(dep, dtype='float64')
                dep_scaled = dep * scalefactor
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
        print('amount must be even')
        return
    else:
        if scaled:
            if energy_flag:
                dataset_name = 'dataset_energy_csv_scaled'
            else: 
                dataset_name = 'dataset_csv_scaled'
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
            print('Deposit:')
            #print(dep)
            print('Max value: '+str(np.max(dep)) + ', Min value: '+str(np.min(dep)))
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

if __name__ == "__main__":
    convert_root_to_csv(os.getcwd())
    findMaxMin(os.getcwd())
    splitTrainTest(os.getcwd(), scaled=True)
    scaleDataset(os.getcwd())
    plotExample(os.getcwd(), amount=6)