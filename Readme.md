# Dataset description
File names beginning with _e_ refer to electrons, while file names beginning with _p_ refer to protons.

# Energies
Files are divided by the total energy released by the calorimeter (N.B. this energy isn't necessarily the initial energy of the particle).

The files are divided in logaritmic bins, therefore the ratio between the energy of the i-th bin and the (i+1)-th bin is fixed.
The only execption is the first _0-100_ bin where all the events with less than 100 GeV can be found.

# File contents
Each file contains a _tree_ called `showers` that is divided in 3 _branch_:
- id: progressive number identifing the event globally (for all the energy level of that particular particle),
- E0: initial energy of the particle,
- dep: array with a fraction of mean energy deposit in a particular bin of `t` and `r` , the longitudinal and transversal coordinates of the swarm.
Each tree can be thought of as a table and each branch as a column, while each row corrisponds to an event.

# Deposit characteristics
Each array has 400 elements, representing an unraveled 20x20 matrix.

The deposit values in each cell are normalized to the total deposit of the event and should therefore be always between 0 and 1.

# Accessing data
The python library `uproot` can be used to import each `tree` as a `numpy.array` or a `pandas.Dataframe`, as shown in the following snippet: 
```
import uproot as up
import numpy as np

e_file = up.open("path/enebins/e_100-126_eventTables.root")
dep = ef["showers"]["dep"].array(library="np")

dep0 = dep[0].reshape(20,20) 
...
```

# Directory Structure

## utils.py

Containes some preliminary functions:

- convert_root_to_csv(data_root, energy_flag=False) opens the root files inside the directory _data\_root_/_enebins_ and writes the data in a csv file inside _data\_root_/_data_. _energy\_flag_ is used to choose if the energy deposit has to be included in the csv file. Some checks are performed to ensure that the data is consistent, like only accepting positive energy and 20x20 grids)

- splitTrainTest(data_root, energy_flag=False, size=10626, mod=8) splits the csv file contained in the directory _data\_root_/_data_ in three different csv files to be used for training, validation and testing. _energy\_flag_ is used to choose one of the two different datasets "styles", _size_ and _mod_ are used in the splitting process

- plotExample(data_root, amount, energy_flag=False, size=10627) plots an _amount_ of items from the dataset to test if the conversion process was successful

## dataset.py 

Opens the data and serves it to the DataLoader

- \_\_init\_\_(self, data_root=os.getcwd(), type='train') is the constructor, _type_ is used to discriminate between the DataLoader of the train, validation or test data. At this stage the data is loaded as a string.

- \_\_getitem\_\_(self, index) takes the _index_ line inside the dataset and converts it into two torch Tensors, the first contains the label, the second the data

## net.py

Defines the neural network topology

- \_\_init\_\_(self, type) is the constructor, _type_ is used to choose between two different structures, either `convolutional` or `fully connected`

- forward(self, x) defines how the data _x_ moves through the neural network

## solver.py

## start.py

Initializes all the variables needed for the training process and starts the process