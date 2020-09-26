from moonlander import Agent 
from moonlander import ReplayBuffer
from plots_moon import plotLearning
import numpy as np 
import gym 

if __name__ == '__main__': 
    env = gym.make('LunarLander-v2')
    n_games = 300 
    agent = Agent(gamma=0.99, epsilon=1, lr=1e-3, input_dims=[8], epsilon_dec=1e-3, mem_size=100000, batch_size=64, eps_end=0.01, fc1_dims=128, fc2_dims=128, replace=100, n_actions=4)
    scores = []
    eps_history = []

    for i in range(n_games): 
        done = False 
        score = 0 
        observation = env.reset()
        while not done: 
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            env.render()
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(score)
        
        avg_score = np.mean(scores[-100:])
        print('Episode ', i, 'score %.1f' % score, 'average score %1.f' % avg_score, 'epsilon %2.f' % agent.epsilon)

    filename = 'lunarlander.png'

    x = [i+1 for i in range(n_games)]
    plotLearning(x, scores, eps_history, filename)