# Slime environment for MARL

This project is a porting of [NetLogo "Slime" simulation model](http://www.netlogoweb.org/launch#http://ccl.northwestern.edu/netlogo/models/models/Sample%20Models/Biology/Slime.nlogo) to Python, and to Farama Foundation [Gymnasium](https://github.com/Farama-Foundation/Gymnasium).
The goal is to make such model available to 3rd party (MA)RL libraries such as [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) and Ray [RLlib](https://github.com/ray-project/ray).
The motivation is to **experiment with (MA)RL applied to communication actions for achieving coordination** amongst agents.

# Project structure

The project is under development, hence **everything is provisional** and subject to change, nebertheless any meaningful change will be reported here.
The most advanced development branch is `sm-baselines-api`, where the single agent environment is compatible with Gym (still need to check Gymnasium) and on its way to be compatible with stable-baselines3.

There, the project is structured as follows:

```
slime_environments
|__environments
   |__SlimeEnvSingleAgent.py         # single agent learning environment
   |__SlimeEnvMultiAgent.py          # multi-agent learning environment
|__agents
   |__MA_QLearning.py                # independent Q-learning
   |__SA_QLearning.py                # single agent Q-learning
   |__multi-agent-params.json        # multi-agent environment params
   |__single-agent-params.json       # single agent environment params
   |__ma-learning-params.json        # multi-agent learning params
   |__sa-learning-params.json        # single agent learning params
```   
