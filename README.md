

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


pip install -e /Users/dwane/projects/NACE_clean


# Documentation

 - See ./writeup/post.md
 - ICLR Blogpost on bsuite which introduces bsuite-applications https://iclr-blogposts.github.io/2023/blog/2023/bsuite-applications/
 - bsuite overview https://github.com/google-deepmind/bsuite
 - bsuite environment descriptions https://github.com/google-deepmind/bsuite/blob/main/bsuite/analysis/results.ipynb
 - via farama: https://shimmy.farama.org/environments/bsuite/
 - list of 23 environments: https://github.com/google-deepmind/bsuite/blob/main/bsuite/bsuite.py which are variants of the 9 games: bandit, cartpole, catch, deep_sea, mnist, mountain_car, umbrella, memory and discounting.
 - environments
   - bandit A simple independent-armed bandit problem. 
     - The agent is faced with 11 actions with deterministic rewards [0.0, 0.1, .., 1.0] randomly assigned. 
     - Run over 20 seeds for 10k episodes. 
     - Score is 1 - 2 * average_regret at 10k episodes. 
     - Must log episode, total_regret for standard analysis.
   - catch DeepMind's internal "hello world" for RL agents. 
     - The environment is a 5x10 grid with a single falling block per episodes (similar to Tetris). 
     - The agent controls a single "paddle" pixel that it should use to "catch" the falling block. 
     - If the agent catches the block reward +1, if the agent misses the block reward -1. 
     - Run the agent for 10k episodes and 20 seeds. 
     - Score is percentage of successful "catch" over first 10k episodes. 
     - Must log episode, total_regret for standard analysis.
   - deepsea - Scalable chain domains that test for deep exploration. 
     - The environment is an N x N grid with falling blocks similar to catch. However the block always starts in the top left. 
     - In each timestep, the agent can move the block "left" or "right". At each timestep, there is a small cost for moving "right" and no cost for moving "left". 
     - However, the agent can receive a large reward for choosing "right" N-times in a row and reaching the bottom right. 
     - This is the single rewarding policy, all other policies receive zero or negative return making this a very difficult exploration problem.
   - discounting_chain
   - memory_chain
   - mnist The "hello world" of deep learning, now as a contextual bandit. 
     - Every timestep the agent must classify a random MNIST digit. 
     - Reward +1 for correct, -1 for incorrect. 
     - Run for 10k episodes, 20 seeds. 
     - Score is percentage of successful classifications. 
     - Must log episode, total_regret for standard analysis.
   - umbrella_chain - A stylized problem designed to highlight problems to do with temporal credit assignment and scaling with time horizon. 
     - The state observation is [need_umbrella, have_umbrella, time_to_go,] + n "distractor" features that are iid Bernoulli. 
     - At the start of each episode the agent observes if it will need an umbrella. 
     - It then has the chance to pick up an umbrella only in the first timestep. 
     - At the end of the episode the agent receives a reward of +1 if it made the correct choice of umbrella, but -1 if it made the incorrect choice. 
     - During chain_length intermediate steps rewards are random +1 or -1.
   - mountain_car A classic benchmark problem in RL. The agent controls an underpowered car and must drive it out of a valley. 
     - Reward of -1 each step until the car reaches the goal. 
     - Maximum episode length of 1000 steps. 
     - Run 1000 episodes for 20 seeds. 
     - Score is based on regret against "good" policy that solves in 25 steps. 
     - Must log episode, total_regret for standard analysis.
   - cartpole A classic benchmark problem in RL. The agent controls a cart on a frictionless plane. 
     - The poles starts near-to upright. 
     - The observation is [x, x_dot, sin(theta), sin(theta)_dot, cos(theta), cos(theta)_dot, time_elapsed]
     - Episodes end once 1000 steps have occured, or |x| is greater than 1. 
     - Reward of +1 when pole > 0.8 height. 
     - Run 1000 episodes for 20 seeds. 
     - Score is percentage of timesteps balancing the pole. 
     - Must log episode, total_regret for standard analysis.
   - bandit noise A simple independent-armed bandit problem. 
     - The agent is faced with 11 actions with deterministic rewards [0.0, 0.1, .., 1.0] randomly assigned. 
     - Run noise_scale = [0.1, 0.3, 1., 3, 10] for 4 seeds for 10k episodes. 
     - Score is 1 - 2 * average_regret at 10k episodes. 
     - Must log episode, total_regret for standard analysis.


