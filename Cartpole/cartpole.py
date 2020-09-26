import gym 
import random 

env_name = "CartPole-v1"
env = gym.make(env_name)

class Agent (): 
    def __init__(self, env): 
        self.action_size = env.action_space.n
        print("Action Size: ", self.action_size)

    def get_action (self): 
        #action = random.choice(range(self.action_size))
        pole_angle = state[2]
        action = 0 if pole_angle < 0 else 1
        return action 


agent = Agent(env)
state = env.reset()

for _ in range(1000): 
    #action = agent.action_space.sample()
    action = agent.get_action()
    state, reward, done, info = env.step(action)
    env.render()