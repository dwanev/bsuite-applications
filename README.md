

This project is a fork of (), being used to investigate different RL algorithms.


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
