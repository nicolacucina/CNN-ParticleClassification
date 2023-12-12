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

# 
dep0 = dep[0].reshape(20,20) 
...
```
