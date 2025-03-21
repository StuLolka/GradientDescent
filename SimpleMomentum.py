import numpy as np
import matplotlib.pyplot as plt


# Function and its derivative
def func(x):
    return -0.5 * x + 0.2 * x ** 2 - 0.01 * x ** 3 - 0.3 * np.sin(4 * x)


def df(x):
    return -0.5 + 2 * 0.2 * x - 3 * 0.01 * x ** 2 - 1.2 * np.cos(4 * x)


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
eta = 0.1
N = 100
xx_1 = -3.5
xx_2 = -3.5
gamma = 0.8
v = 0

plt.ion()
fig, ax = plt.subplots()

for i in range(N):
    # Gradient descent with momentum
    v = gamma * v + (1 - gamma) * eta * df(xx_1)
    xx_1 -= v

    # Gradient descent without momentum
    v_2 = eta * df(xx_2)
    xx_2 -= v_2

    redraw_function(ax, i, np.min([xx_1, xx_2]), np.max([xx_1, xx_2]))
    draw(ax, xx_1, 'green', 'Momentum')
    draw(ax, xx_2, 'red', 'Simple')

    plt.pause(0.1)
    fig.canvas.draw()

plt.ioff()
plt.show()