#Exercise 3.11 - Modified Example 3.15 Linear Regression
import matplotlib
matplotlib.use('TkAgg')  # force a GUI backend so plt.show() opens a window
import matplotlib.pyplot as plt
from scipy import stats

# More data points in x and y
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
y = [3, 5, 5, 6, 7, 8, 8, 9, 10, 12, 11, 13, 14, 15, 16]

slope, intercept, r, p, std_err = stats.linregress(x, y)
print("slope: ", slope)
print("intercept: ", intercept)

def myfunc(x):
    return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y, label="Data Points")
plt.plot(x, mymodel, color="red", label="Regression Line")

plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.title("Linear Regression Example")
plt.legend()
plt.grid(True)

plt.savefig("linreg_plot.png", dpi=150)  # backup: saved next to this script
plt.show(block=True)
