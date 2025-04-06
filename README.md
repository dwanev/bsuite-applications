

This project is a fork of (), being used to investigate different RL algorithms.

# Status:

It is a work in progress, runs and invoke NACE in deepsea, but fails due to limitations in NACE. 
Specifically i) when nothing changes, no rules are added or removed. (i.e.e does not learn can not move into a wall)
ii) context could be injected into algo so that it can be more sample efficient between episodes.
iii) fails to explore RHS as quickly as it could/should uncertain as to why.


## Installation
Use python 3.6>= x <9
```
python3 -m venv venv_bsuite_applications
source venv_bsuite_applications/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m pip install nace


```
You may need to install tensorflow separately as well.




## Running experiments
```
python run.py --help
or  
python run.py -e 1.1 -o ./tmp3 --overwrite T  
```

## Generating plots
```
python analyze.py --help
or
python analyze.py -e 1.1 -i ./tmp3  


# Experiment combinations are defined in experiment_definitions.py  i.e. for 1.1 and the code for them in model_configs.py  
   
  
```


## Installing my Agent code from a local directory

Note: if any code is moved, this will break, but means I'll use the latest versions

pip install -e ../TX-Jupiter-Notebooks/0040_NACE_clean