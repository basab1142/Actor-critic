import numpy as np
import torch
from actor_critic_implementaion import *
from environment import *

agent = Agent(n_actions=3, feature_space= 4)

score_history = []
succes_history = []
success = 0 
n_games = 30000
for episode in range(1, n_games + 1):
    env = CATCHTHEBALL(10)
    observation = np.array(env.getFeature()[0], dtype = np.float32)
    done = False
    score = 0
    max_step = 100

    while not done and max_step:
        action = agent.action_selection(observation)
        env.movePlate(action)
        env.ballMove()
        observation_, reward, done = env.getFeature()
        score += reward
        observation = np.array(observation_, dtype = np.float32)
        max_step -= 1
        if reward == 1:
            success += 1
    score_history.append(score)
    succes_history.append(success)
    avg_score = np.mean(score_history[-100:])
    
    if episode % 100== 0:
        agent.optimizer = torch.optim.Adagrad(params=agent.network.parameters(), lr=agent.lr*0.9999)
        print(f'episode  {episode} score  {score} avg_score  {avg_score:.2f} success {success}')
