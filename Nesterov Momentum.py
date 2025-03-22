import numpy as np
import matplotlib.pyplot as plt


# Function and its derivative
def func(x):
    return 0.4 * x + 0.1 * np.sin(2*x) + 0.2 * np.cos(3*x)


def df(x):
    return 0.4 + 0.2 * np.cos(2*x) - 0.6 * np.sin(3*x)


def redraw_function(ax, iteration, x_min, x_max):
    ax.clear()
    ax.set_title(f"Iteration №{iteration + 1}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    x_min -= 5
    x_max += 5
    x = np.arange(x_min, x_max, 0.01)
    ax.plot(x, func(x), color='blue', label="f(x)")


def draw(ax, xx, color, label):
    ax.scatter(xx, func(xx), color=color, label=f"{label} ({xx:.2f}, {func(xx):.2f})")
    ax.legend()


# Gradient descent parametrs
xx_1 = xx_2 = xx_3 = 4
v_1 = v_2 = v_3 =  0
eta = 1
N = 100
gamma = 0.7

plt.ion()
fig, ax = plt.subplots()

for i in range(N):
    # Gradient descent without momentum
    v_1 = eta * df(xx_1)
    xx_1 -= v_1

    # Gradient descent with momentum
    v_2 = gamma * v_2 + (1 - gamma) * eta * df(xx_2)
    xx_2 -= v_2

    # Gradient descent with Nesterov momentum
    v_3 = gamma * v_3 + (1 - gamma) * eta * df(xx_3 - gamma * v_3)
    xx_3 -= v_3

    redraw_function(ax, i, np.min([xx_1, xx_2, xx_3]), np.max([xx_1, xx_2, xx_3]))
    draw(ax, xx_1, 'red', 'Simple')
    draw(ax, xx_2, 'orange', 'Momentum')
    draw(ax, xx_3, 'green', 'Nesterov momentum')

    plt.pause(0.1)
    fig.canvas.draw()

plt.ioff()
plt.show()