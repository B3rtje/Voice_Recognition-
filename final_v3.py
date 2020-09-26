import gym 
import tensorflow as tf 
import pylab
from tensorflow import keras
import random 
import numpy as np 
import datetime as dt
from plots_moon import plotLearning
import pandas as pd

#Hyperparameters: 
STORE_PATH = 'Space_invaders_v0'
max_eps = 1
min_eps = 0.1

#Usually epsilon is set arround 0.1 aka 10
#Gamma is the discount rate Between 1 and 0 the higher the value the less you are discounting
gamma = 0.99
batch_size = 64
TAU = 0.08
post_pro_img_size = (84, 84, 0)
number_frames = 4
DELAY_TRAINING = 1000000
EPSILON_MIN_ITER = 500000

env = gym.make('SpaceInvaders-v4')
number_actions = env.action_space.n 

class KerasModel(keras.Model):
    def __init__(self, hidden_size: int, number_actions: int, dueling: bool): 
        super(KerasModel, self).__init__()
        self.dueling = dueling
        self.convolutional1 = keras.layers.Conv2D(16, (8, 8), (4, 4), activation='relu')
        self.convolutional2 = keras.layers.Conv2D(32, (4, 4), (2, 2), activation='relu')
        self.flatten = keras.layers.Flatten()
        #He normal = It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / fan_in) where fan_in is the number of input units in the weight tensor.
        self.adv_dense = keras.layers.Dense(hidden_size, activation='relu', kernel_initializer=keras.initializers.he_normal())
        self.adv_out = keras.layers.Dense(number_actions, kernel_initializer=keras.initializers.he_normal())

        #Just the boolean element (dueling see above!)        
        if dueling: 
            self.v_dense = keras.layers.Dense(hidden_size, activation='relu', kernel_initializer=keras.initializers.he_normal())
            self.v_out = keras.layers.Dense(1, kernel_initializer=keras.initializers.he_normal())
            #Computes the mean of elements across dimensions of a tensor.
            self.lambda_layer = keras.layers.Lambda(lambda x: x - tf.reduce_mean(x))
            # Layer that adds a list of inputs. It takes as input a list of tensors, all of the same shape, and returns a single tensor (also of the same shape)
            self.combine = keras.layers.Add()

    def call(self, input):
        x = self.convolutional1(input)
        x = self.convolutional2(x)
        x = self.flatten(x)
        adv = self.adv_dense(x)
        adv = self.adv_out(adv)
        if self.dueling:
            v = self.v_dense(x)
            v = self.v_out(v)
            norm_adv = self.lambda_layer(adv)
            combined = self.combine([v, norm_adv])
            return combined
        return adv

#Hidden size you can pick a number: 16, 32, 64, 128, 256 --> most common is 256 
# After the model we gonna make 2 models (head & target), Both of these networks will be utilised in the Double Q component of the learning
# The target_network weights are then set to be initially equeal to the prmary_networks weights.
# Finaly head network is compiled for training with Adam optim. & loss function MSE (Mean squared error). 
head_network = KerasModel(256, number_actions, True)
target_network = KerasModel(256, number_actions, True)
#mse is beter voor de outliners te leren kennen... 
head_network.compile(optimizer=keras.optimizers.Adam(), loss='mse')

for t, e in zip(target_network.trainable_variables, head_network.trainable_variables): 
    t.assing(e)

# Huber loss is een combinatie between MSE & MAE(=Mean Asolute Error) --> MAE is beter om outliners te negeren. 
head_network.compile(optimizer=keras.optimizers.Adam(), loss=tf.keras.losses.Huber())

#Class memory I create because of improve network stability and make sure previous experiences are not discarded but used in training
class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._actions = np.zeros(max_memory, dtype=np.int32)
        self._rewards = np.zeros(max_memory, dtype=np.float32)
        self._frames = np.zeros((post_pro_img_size[0], post_pro_img_size[1], max_memory), dtype=np.float32)
        self._terminal = np.zeros(max_memory, dtype=np.bool)
        #Just a counter variable: --> to record the present location of stored samples in the mem buffer and it ensure that the memory is noto overflowed. 
        self.counter_var = 0

#Batches of experiences are randomly sampled from memory and are used to train the neural network
    def add_sample(self, frame, action, reward, terminal):
        self._actions[self.counter_var] = action
        self._rewards[self.counter_var] = reward
        self._frames[:, :, self.counter_var] = frame[:, :, 0]
        self._terminal[self.counter_var] = terminal
        if self.counter_var % (self._max_memory - 1) == 0 and self.counter_var != 0:
            self.counter_var = batch_size + number_frames + 1
        else:
            self.counter_var += 1

    def sample(self):
        if self.counter_var < batch_size + number_frames + 1:
            raise ValueError("Not enough memory to extract a batch")
        else:
            max_mem = min(self.counter_var, self._max_memory)
            batch = np.random.choice(max_mem, batch_size, replace=False)
            random_index = np.random.randint(number_frames + 1, self.counter_var, size=batch_size)
            states = np.zeros((batch_size, post_pro_img_size[0], post_pro_img_size[1], number_frames),
                             dtype=np.float32)
            next_states = np.zeros((batch_size, post_pro_img_size[0], post_pro_img_size[1], number_frames),
                             dtype=np.float32)
            for i, idx in enumerate(random_index):
                states[i] = self._frames[:, :, idx - 1 - number_frames:idx - 1]
                next_states[i] = self._frames[:, :, idx - number_frames:idx]
            return states, self._actions[batch], self._rewards[batch], next_states, self._terminal[batch]

#Memory you can just pick a number! Here we fit the class! 
memory = Memory(1000000)

def image_preprocess(image, new_size=(84, 84)):
    # convert to greyscale, resize and normalize the image
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, new_size)
    image = image / 255
    return image

#Value of eps = Max + nakijken v/d else function! 
def choose_action(state, primary_network, eps, step):
    if random.random() < eps:
        return random.randint(0, number_actions - 1)
    else:
        return np.argmax(primary_network(tf.reshape(state, (1, post_pro_img_size[0], post_pro_img_size[1], number_frames)).numpy()))

#This function slowly shift the target network weights towards the primary network weights in accordance with the Double Q learning methodology. 
def update_network(primary_network, target_network):
    for t, e in zip(target_network.trainable_variables, primary_network.trainable_variables):
        t.assign(t * (1 - TAU) + e * TAU)

#input = existing state stack + newest state to be added, then It shuffles all the existing frames within the state_stack “back” one position.
# Finally, the most recent state or frame is stored in the newly vacated row 2 of the state stack.
def process_state_stack(state_stack, state):
    for i in range(1, state_stack.shape[-1]):
        state_stack[:, :, i - 1].assign(state_stack[:, :, i])
    state_stack[:, :, -1].assign(state[:, :, 0])
    return state_stack

def train(head_network, memory, target_network=None):
    states, actions, rewards, next_states, terminal = memory.sample()
    # predict Q(s,a) given the batch of states
    prim_qt = head_network(states)
    # predict Q(s',a') from the evaluation network
    prim_qtp1 = head_network(next_states)
    # copy the prim_qt tensor into the target_q tensor - we then will update one index corresponding to the max action
    target_q = prim_qt.numpy()
    updates = rewards
    valid_idxs = terminal != True
    batch_idxs = np.arange(batch_size)
    if target_network is None:
        #np.amax == finds the value of maximum element in that entire array 
        updates[valid_idxs] += gamma * np.amax(prim_qtp1.numpy()[valid_idxs, :], axis=1)
    else:
        # np.argmax == Returns the indices of the maximum values along an axis
        prim_action_tp1 = np.argmax(prim_qtp1.numpy(), axis=1)
        q_from_target = target_network(next_states)
        updates[valid_idxs] += gamma * q_from_target.numpy()[batch_idxs[valid_idxs], prim_action_tp1[valid_idxs]]
    target_q[batch_idxs, actions] = updates
    loss = head_network.train_on_batch(states, target_q)
    return loss

scores = []
episodes = []
 
num_episodes = 2000000
eps = max_eps
render = False
train_writer = tf.summary.create_file_writer(STORE_PATH + f"/DuelingQSI_{dt.datetime.now().strftime('%d%m%Y%H%M')}")
double_q = True
steps = 0
for i in range(num_episodes):
    done = False
    state = env.reset()
    state = image_preprocess(state)
    state_stack = tf.Variable(np.repeat(state.numpy(), number_frames).reshape((post_pro_img_size[0], post_pro_img_size[1], number_frames)))
    cnt = 1
    avg_loss = 0
    tot_reward = 0
    while not done:
        if render:
            env.render()
        action = choose_action(state_stack, head_network, eps, steps)
        next_state, reward, done, info = env.step(action)
        env.render()
        tot_reward += reward if not done or tot_reward > 150 else -100
        next_state = image_preprocess(next_state)
        state_stack = process_state_stack(state_stack, next_state)
        #Very important because a lot of programmers forget this!! --> set state to the next state!!
        state = next_state
        # store in memory
        memory.add_sample(next_state, action, reward, done)

        if steps > DELAY_TRAINING:
            loss = train(head_network, memory, target_network if double_q else None)
            update_network(head_network, target_network)

            score = tot_reward if tot_reward > 150 else tot_reward + 100
            scores.append(score)
            episodes.append(i)

            #if np.mean(scores[-min(10, len(scores)):]) > 1000:
            #    print("System has improved!")
            #if np.mean(scores[-min(10, len(scores)):]) > 1500:
            #    print("Stage II improvement")
            #if np.mean(scores[-min(10, len(scores)):]) > 2000:
            #    print("Stage 3 improvement")
        else:
            loss = -1
        avg_loss += loss

        # linearly decay the eps value
        if steps > DELAY_TRAINING:
            eps = max_eps - ((steps - DELAY_TRAINING) / EPSILON_MIN_ITER) * (max_eps - min_eps) if steps < EPSILON_MIN_ITER else min_eps
        steps += 1

        if done:
            if steps > DELAY_TRAINING:
                avg_loss /= cnt
                if i % 5 == 0 : 
                    data_for_panda = {"Scores": scores, "Episodes": episodes}
                    dataframe = pd.DataFrame(data_for_panda)
                    dataframe.plot.line(x='Episodes', y='Scores').get_figure().savefig('/Users/bert/Desktop/vg_games_ai_final/Grafieken_final/output_v3.png')

                print(f"Episode: {i}, Reward: {tot_reward}, avg loss: {avg_loss:.5f}, eps: {eps:.3f}")
                with train_writer.as_default():
                    tf.summary.scalar('reward', tot_reward, step=i)
                    tf.summary.scalar('avg loss', avg_loss, step=i)
            else:
                print(f"Pre-training...Episode: {i}", "total of steps: ", steps, " reward of the game: ", tot_reward )

            break
        
        cnt += 1