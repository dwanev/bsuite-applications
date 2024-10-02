

This project is a fork of (), being used to investigate different RL algorithms.

# Status:

It is a work in progress, and does not yet run.

## Installation
Use python 3.6>= x <9
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
You may need to install tensorflow separately as well.




## Running experiments
```
python run.py --help
or  
python run.py -e 1.5 -o ./tmp3 --overwrite T  
```

## Generating plots
```
python analyze.py --help
or
python analyze.py -e 1.5 -i ./tmp3   
  
```


## Installing my Agent code from a local directory

Note: if any code is moved, this will break, but means I'll use the latest versions

pip install -e ../TX-Jupiter-Notebooks/0040_NACE_clean