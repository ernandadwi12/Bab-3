#Exercise 3.15 - Modified Example 3.21: Semi-supervised Learning with 3 groups
# Added a third group of points. Only ONE point per group is labeled.
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.semi_supervised import LabelSpreading

# Group 0 (bottom-left), Group 1 (top-right, original), Group 2 (new third group)
X = np.array([
    [0, 1], [1, 1], [2, 0], [3, 1],        # Group 0
    [10, 5], [11, 6], [12, 4], [13, 5],    # Group 1
    [5, 12], [6, 13], [4, 11], [6, 11]     # Group 2 (new)
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

# Only ONE point per group is labeled; everything else is unlabeled (-1)
labels = np.full(12, -1.)
labels[0] = 0    # one labeled point in Group 0
labels[4] = 1    # one labeled point in Group 1
labels[8] = 2    # one labeled point in Group 2
print("Initial labels:")
print(labels)

label_spread = LabelSpreading(kernel='knn', n_neighbors=3, alpha=0.8, max_iter=1000)
label_spread.fit(X, labels)
output_labels = label_spread.transduction_
print("\nPropagated labels:")
print(output_labels)

# Visualize the three groups and the propagated labels
plt.figure(figsize=(7, 5))
color_map = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green'}
colors = [color_map[lbl] for lbl in output_labels]
plt.scatter(X[:, 0], X[:, 1], c=colors, s=100, edgecolor='k')

# Mark the three originally-labeled points
labeled_idx = [0, 4, 8]
label_colors = ['red', 'purple', 'black']
for idx, c in zip(labeled_idx, label_colors):
    plt.scatter(X[idx, 0], X[idx, 1], facecolor='none', edgecolor=c,
                s=300, linewidths=2, label=f'Labeled point (class {int(y[idx])})')

for i, (px, py) in enumerate(X):
    plt.annotate(str(i), (px, py), textcoords="offset points", xytext=(5, 5))

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Semi-supervised Label Spreading (3 groups, 1 labeled point each)")
plt.legend()
plt.grid(True)
plt.savefig("semisup3_plot.png", dpi=150)
plt.show(block=True)