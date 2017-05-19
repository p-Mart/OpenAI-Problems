import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

#OpenAI Gym code.
env = gym.make('CartPole-v0')

class Memory(object):
	def __init__(self, s, a, s_):
		self.s = s
		self.a = a
		self.s_ = s_

class Experience(object):
	def __init__(self):
		self.experience = deque([])
		self.size = 500
		self.batch_size = 50

	def storeMemory(self, m):
		assert m.__class__.__name__ == "Memory"

		if(self.size >= 500):
			self.experience.popleft()
			self.size -= 1

		self.experience.append(m)
		self.size += 1

	def returnBatch(self):
		s = np.array((self.batch_size, 4))
		a = np.array((self.batch_size, 2))
		s_ = np.array((self.batch_size, 4))

		for i in range(self.batch_size):
			rand = random.randint(self.size)
			s[i] = self.experience[rand].s
			a[i] = self.experience[rand].a
			s_[i] = self.experience[rand].s_

		return s, a, s_




'''
class ActorNetwork(object):

	def __init__(self, sess, learning_rate):

		NODES_H1 = 2
		NODES_H2 = 2

		self.sess = sess
		self.learning_rate = learning_rate

		self.keep_prob = tf.placeholder(tf.float32)

		self.x = tf.placeholder(tf.float32, [None, 4])

		W_h1 = tf.Variable(tf.truncated_normal([4, NODES_H1], stddev=0.1))
		b_h1 = tf.Variable(tf.constant(0.1, shape=[NODES_H1]))
		h1 = tf.nn.relu(tf.matmul(self.x, W_h1) + b_h1)
		h1_drop = tf.nn.dropout(h1, self.keep_prob)

		W_h2 = tf.Variable(tf.truncated_normal([NODES_H1, NODES_H2], stddev=0.1))
		b_h2 = tf.Variable(tf.constant(0.1, shape=[NODES_H2]))
		h2 = tf.nn.relu(tf.matmul(h1_drop, W_h2) + b_h2)
		h2_drop = tf.nn.dropout(h2, self.keep_prob)

		W_out = tf.Variable(tf.truncated_normal([NODES_H2, 2], stddev=0.1))
		b_out = tf.Variable(tf.constant(0.1, shape=[2]))
		self.output = tf.matmul(h2_drop, W_out) + b_out

		self.get_action = tf.argmax(tf.nn.softmax(self.output), 1)

	def getOutput(self):
		return self.output

	def getAction(self, observation):
		return self.sess.run(self.get_action, feed_dict={
			self.x : np.array(observation).reshape((1,4)),
			self.keep_prob : 1.0})
'''

def debug():
	print env.action_space
	print env.observation_space

	action = env.action_space.sample()
	observation, reward, info, done = env.step(action)

	print "Action ", action
	print "Observation ", observation
	print "Reward ", reward
	print "Info ", info

def toLogit(x, len):
	logit = [0 for _ in range(len)]
	logit[x] = 1
	return logit

def run():

	epsilon = 0.01 #Exploration probability
	max_frames = 200
	frame_cutoff = 20
	episodes = 20000
	NODES_H1 = 2
	NODES_H2 = 2
	learning_rate = 1e-3

	sess = tf.InteractiveSession()
	###########Actor Network#############
	
	x = tf.placeholder(tf.float32, [None, 4])
	keep_prob = tf.placeholder(tf.float32)

	W_h1 = tf.Variable(tf.truncated_normal([4, NODES_H1], stddev=0.1))
	b_h1 = tf.Variable(tf.constant(0.1, shape=[NODES_H1]))

	h1 = tf.nn.relu(tf.matmul(x, W_h1) + b_h1)
	h1_drop = tf.nn.dropout(h1, keep_prob)


	W_h2 = tf.Variable(tf.truncated_normal([NODES_H1, NODES_H2], stddev=0.1))
	b_h2 = tf.Variable(tf.constant(0.1, shape=[NODES_H2]))

	h2 = tf.nn.relu(tf.matmul(h1_drop, W_h2) + b_h2)
	h2_drop = tf.nn.dropout(h2, keep_prob)

	W_out = tf.Variable(tf.truncated_normal([NODES_H2, 2], stddev=0.1))
	b_out = tf.Variable(tf.constant(0.1, shape=[2]))

	output = tf.matmul(h2_drop, W_out) + b_out
	get_action = tf.argmax(tf.nn.softmax(output), 1)
	
	#actor = ActorNetwork(sess, learning_rate)

	###########Target Network#############
	x_t = tf.placeholder(tf.float32, [None, 4])
	gamma = tf.constant(0.5)

	W_h1_t = tf.Variable(tf.truncated_normal([4, NODES_H1], stddev=0.1))
	b_h1_t = tf.Variable(tf.constant(0.1, shape=[NODES_H1]))

	h1_t = tf.nn.relu(tf.matmul(x_t, W_h1_t) + b_h1_t)
	h1_drop_t = tf.nn.dropout(h1_t, keep_prob)


	W_h2_t = tf.Variable(tf.truncated_normal([NODES_H1, NODES_H2], stddev=0.1))
	b_h2_t = tf.Variable(tf.constant(0.1, shape=[NODES_H2]))

	h2_t = tf.nn.relu(tf.matmul(h1_drop_t, W_h2_t) + b_h2_t)
	h2_drop_t = tf.nn.dropout(h2_t, keep_prob)

	W_out_t = tf.Variable(tf.truncated_normal([NODES_H2, 2], stddev=0.1))
	b_out_t = tf.Variable(tf.constant(0.1, shape=[2]))

	target = tf.scalar_mul(gamma, (tf.matmul(h2_drop_t, W_out_t) + b_out_t))
	get_action_t = tf.argmax(tf.nn.softmax(target), 1)

	#target = tf.placeholder(tf.float32, [None, 2])


	loss = tf.reduce_mean(tf.square(target - output))

	
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	
	sess.run(tf.global_variables_initializer())

	tf.get_default_graph().finalize()

	#Training loop
	for episode in range(episodes):
		
		observation = env.reset()

		#Reduce rate of positive reinforcement over time
		#if (episode+1) % 500 == 0 and frame_cutoff != max_frames:
		#	frame_cutoff += 1

		experience_x = []
		experience_y = []

		for frame in range(max_frames):
			#env.render()

			#Take an action
			if (random.random() < epsilon):
				action = env.action_space.sample()
				observation, reward, done, info = env.step(action)
			else:
				
				action = get_action.eval(feed_dict={
							x:np.array(observation).reshape((1,4)), 
							keep_prob:1.0})
				
				#action = get_action(observation)
				action = int(action[0])
				observation, reward, done, info = env.step(action)

			#Update memory
			experience_x.append(observation)
			experience_y.append(toLogit(action, 2))

			#Check for done flag
			if not done:
				if (frame+1) % frame_cutoff == 0:
					#Train if memory full
					train_step.run(feed_dict={
						x : np.array(experience_x),
						x_t: np.array(experience_x),
						keep_prob : 0.5})
					
					experience_x = []
					experience_y = []
			else:
				#Negative reinforcement

				if(frame < frame_cutoff):
					for i in range(len(experience_y)):
						for j in range(len(experience_y[i])):
							if experience_y[i][j] == 0:
								experience_y[i][j] = 1
							else:
								experience_y[i][j] = 0

					train_step.run(feed_dict={
							x : np.array(experience_x), 
							x_t: np.array(experience_x),
							keep_prob : 0.5})

					experience_x = []
					experience_y = []
				
				#End episode
				break

		#Modify exploration probability
		if epsilon > 0.01:
			epsilon = epsilon / 1.001
		else:
			epsilon = 0.01

		print "Episode", episode, ", frames", frame


	sess.close()

if __name__ == '__main__':
	#debug()
	run()