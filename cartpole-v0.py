import gym
import tensorflow as tf
import numpy as np
import random

#OpenAI Gym code.
env = gym.make('CartPole-v0')

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



	epsilon = 1.0 #Exploration probability
	max_frames = 100
	frame_cutoff = 20
	episodes = 2000

	#Network architecture
	x = tf.placeholder(tf.float32, [None, 4])
	W = tf.Variable(tf.truncated_normal([4, 2], stddev=0.1))
	b = tf.Variable(tf.constant(0.1, shape=[2]))

	output = tf.nn.relu(tf.matmul(x, W) + b)
	target = tf.placeholder(tf.float32, [None, 2])

	cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output))

	learning_rate = 1e-2
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())

	for episode in range(episodes):
		
		observation = env.reset()

		#Reduce rate of positive reinforcement over time
		if (episode+1) % 200 == 0 and frame_cutoff != max_frames:
			frame_cutoff += 5

		experience_x = []
		experience_y = []

		for frame in range(max_frames):
			env.render()

			#Take an action
			if (random.random() < epsilon):
				action = env.action_space.sample()
				observation, reward, done, info = env.step(action)
			else:
				action = tf.argmax(tf.nn.softmax(output), 1).eval(
					feed_dict={x:np.array(observation).reshape((1,4))})

				action = int(action[0])
				observation, reward, done, info = env.step(action)

			#Check for done flag
			if not done:
				if (frame+1) % frame_cutoff != 0:
					#Update memory
					experience_x.append(observation)
					experience_y.append(toLogit(action, 2))
				else:
					#Train if memory full (lol)
					train_step.run(feed_dict={
						x:np.array(experience_x), target:np.array(experience_y)})
					
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
							x:np.array(experience_x), target:np.array(experience_y)})

					experience_x = []
					experience_y = []
				
				#End episode
				break

		print len(experience_x)

		#Modify exploration probability
		if epsilon > 0.25:
			epsilon = epsilon / 1.01
		else:
			epsilon = 0.1

		print "Episode", episode, ", frames", frame

if __name__ == '__main__':
	#debug()
	run()