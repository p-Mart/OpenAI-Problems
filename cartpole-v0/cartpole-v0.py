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
	max_frames = 200
	frame_cutoff = 20
	episodes = 20000

	#Network architecture
	x = tf.placeholder(tf.float32, [None, 4])
	keep_prob = tf.placeholder(tf.float32)

	W_h1 = tf.Variable(tf.truncated_normal([4, 8], stddev=0.1))
	b_h1 = tf.Variable(tf.constant(0.1, shape=[8]))

	h1 = tf.nn.relu(tf.matmul(x, W_h1) + b_h1)
	h1_drop = tf.nn.dropout(h1, keep_prob)


	W_h2 = tf.Variable(tf.truncated_normal([8, 8], stddev=0.1))
	b_h2 = tf.Variable(tf.constant(0.1, shape=[8]))

	h2 = tf.nn.relu(tf.matmul(h1_drop, W_h2) + b_h2)
	h2_drop = tf.nn.dropout(h2, keep_prob)

	W_out = tf.Variable(tf.truncated_normal([8, 2], stddev=0.1))
	b_out = tf.Variable(tf.constant(0.1, shape=[2]))

	output = tf.matmul(h2_drop, W_out) + b_out

	target = tf.placeholder(tf.float32, [None, 2])

	get_action = tf.argmax(tf.nn.softmax(output), 1)

	cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output))

	learning_rate = 1e-2
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

	sess = tf.InteractiveSession()
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
						target : np.array(experience_y), 
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
							target : np.array(experience_y),
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