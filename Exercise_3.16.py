#Exercise 3.16 - Modified Example 3.22: Custom 8-state routing diagram
#Modified from:
#https://amunategui.github.io/reinforcement-learning/index.html
#http://firsttimeprogrammer.blogspot.com/2016/09/getting-ai-smarter-with-q-learning.html
#http://mnemstudio.org/path-finding-q-learning-tutorial.htm

import numpy as np
import pylab as plt
from Q_Utils import *

# Setting up Parameters =========================================
# Custom 8-state routing diagram (rooms 0 to 7)
# Layout (undirected connections):
#   0 -- 1 -- 2
#        |    \
#        3     4
#         \   /
#          5
#         / \
#        6   7  (7 is the goal)
points_list = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 5), (5, 6), (5, 7)]
goal = 7

# Show the routing graph
showgraph(points_list)

# The number of points of the R matrix (8 states: 0-7)
MATRIX_SIZE = 8

# create matrix R
R = createRmat(MATRIX_SIZE, points_list, goal)

# Create matrix Q
Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))

# Learning parameter
gamma = 0.8

# Training ======================================================
scores = []
for i in range(700):
    # Select a random current_state (starting point)
    current_state = np.random.randint(0, int(Q.shape[0]))
    # Work out all the available next step actions
    available_act = available_actions(R, current_state)
    # Choose a random next step action
    action = sample_next_action(available_act)
    # Update the Q matrix
    score = update(R, Q, current_state, action, gamma)
    scores.append(score)
    print('Score:', str(score))

print("Trained Q matrix:")
print(Q / np.max(Q) * 100)

# Testing =======================================================
current_state = 0
steps = [current_state]
while current_state != goal:
    next_step_index = np.where(Q[current_state, ] == np.max(Q[current_state, ]))[1]
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size=1)[0])
    else:
        next_step_index = int(next_step_index[0])
    steps.append(next_step_index)
    current_state = next_step_index

# Display Results ===============================================
print("Most efficient path:")
print(steps)
plt.plot(scores)
plt.show()