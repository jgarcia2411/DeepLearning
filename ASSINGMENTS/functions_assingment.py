import numpy as np
import matplotlib.pyplot as plt
def cuadraticplot(coefficients = [1,0,0], x_limits = [-20,20]):
    x = np.arange(x_limits[0], x_limits[-1], 1)
    y = coefficients[0]*(x**2) + coefficients[1]*x + coefficients[-1]
    plt.plot(x, y)
    plt.title("Cuadratic Function")
    plt.xlabel("Values of x")
    plt.ylabel("Values of y")
    plt.savefig('cuadratic_plot.png')

cuadraticplot(coefficients = [3,-2,4], x_limits=[-10,10])

def cubicplot(coefficients = [1,0,0,0], x_limits = [-20,20]):
    x = np.arange(x_limits[0], x_limits[-1], 1)
    y = coefficients[0]*(x**3) + coefficients[1]*(x**2) + coefficients[2]*x + coefficients[-1]
    plt.plot(x, y)
    plt.title("Cubic Function")
    plt.xlabel("Values of x")
    plt.ylabel("Values of y")
    plt.savefig('cubic_plot.png')

cubicplot(coefficients = [2,24,-54,0])

