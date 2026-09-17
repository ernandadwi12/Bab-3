#Exercise 3.14 - Modified Example 3.21: Semi-supervised Learning
# Added 2 more data points to each group (group 0 and group 1)
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.semi_supervised import LabelSpreading

# Original points:
#   Group 0: [0,1],[1,1],[2,0],[3,1]
#   Group 1: [10,5],[11,6],[12,4],[13,5]
# Added 2 new points to each group (in bold below):
X = np.array([
    [0, 1], [1, 1], [2, 0], [3, 1], [1, 2], [2, 2],          # Group 0 (6 points, last 2 are new)
    [10, 5], [11, 6], [12, 4], [13, 5], [12, 6], [11, 4]      # Group 1 (6 points, last 2 are new)
])
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Only label the very first point as 0 and the very last point as 1;
# everything else is unlabeled (-1), same idea as the original example
labels = np.full(12, -1.)
labels[0] = 0
labels[-1] = 1
print("Initial labels:")
print(labels)

label_spread = LabelSpreading(kernel='knn', alpha=0.8)
label_spread.fit(X, labels)
output_labels = label_spread.transduction_
print("\nPropagated labels:")
print(output_labels)

# Visualize: show which points were originally labeled vs unlabeled,
# and the final predicted label for every point
plt.figure(figsize=(7, 5))
colors = ['tab:blue' if lbl == 0 else 'tab:orange' for lbl in output_labels]
plt.scatter(X[:, 0], X[:, 1], c=colors, s=100, edgecolor='k')

# Mark the two originally-labeled points
plt.scatter(X[0, 0], X[0, 1], facecolor='none', edgecolor='red', s=300, linewidths=2, label='Labeled point (0)')
plt.scatter(X[-1, 0], X[-1, 1], facecolor='none', edgecolor='green', s=300, linewidths=2, label='Labeled point (1)')

for i, (px, py) in enumerate(X):
    plt.annotate(str(i), (px, py), textcoords="offset points", xytext=(5, 5))

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Semi-supervised Label Spreading (12 points, 2 labeled)")
plt.legend()
plt.grid(True)
plt.savefig("semisup_plot.png", dpi=150)
plt.show(block=True)