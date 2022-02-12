#####################libraries#####################################
import numpy as np
from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

################################################################

def cuadraticplot(coefficients = [1,0,0], x_limits = [-20,20]):
    """Plotting function: return plot of a polynomial degree 2 function
    coefficients : list of function coefficients y= ax^2 +bx +c
    x_limits : plot scale in x axes"""

    x2 = np.arange(x_limits[0], x_limits[-1], 1)
    y2 = coefficients[0]*(x2**2) + coefficients[1]*x2 + coefficients[-1]
    plt.clf()
    plt.plot(x2, y2)
    plt.title("Cuadratic Function")
    plt.xlabel("Values of x")
    plt.ylabel("Values of y")
    plt.savefig('cuadratic_plot.png')
    plt.show()

def cubicplot(coefficients = [1,0,0,0], x_limits = [-20,20]):
    """Plotting function: return plot of a polynomial degree 3 function
        coefficients : list of function coefficients y= ax^3+ bx^2 +cx +d
        x_limits : plot scale in x axes"""
    x3 = np.arange(x_limits[0], x_limits[-1], 1)
    y3 = coefficients[0]*(x3**3) + coefficients[1]*(x3**2) + coefficients[2]*x3 + coefficients[-1]
    plt.clf()
    plt.plot(x3, y3)
    plt.title("Cubic function")
    plt.xlabel("Values of x")
    plt.ylabel("Values of y")
    plt.savefig('cubic_plot.png')
    plt.show()

#################### Q.2, Q.3 Plot ####################################
cubicplot(coefficients = [2,24,-54,0], x_limits= [-20,20])
##################################################################


################### Q.4 3D PLOT ####################################
def z_func(x, y):
    return (x ** 2 + y ** 2)

x = arange(-3.0, 3.0, 0.1)
y = arange(-3.0, 3.0, 0.1)
X, Y = meshgrid(x, y)  # grid of point
Z = z_func(X, Y)  # evaluation of the function on the grid

plt.clf()
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                      cmap=cm.RdBu,linewidth=0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
plt.savefig('3d_plot.png')
########################################################################

##################### E.2 Expanding Vectors ##################
vector = [1,2,2]
v1 = [-1,1,0]
v2 = [1,1,-2]
v3 = [1,1,0]

print('E.2 Solution')
print(f'X = {vector}')
print(f'v1 = {v1}')
print(f'v2 = {v2}')
print(f'v3 = {v3}')
print('checking for orthogonality')
print(f'values of dot product y1,y2={np.dot(v1,v2)} y2,y3={np.dot(v2,v3)} y1,y3={np.dot(v1,v3)}') # orthogonality
print('Obtaining reciprocal basis vectors approach')


# Calculating reciprocal matrix
matrix = [
    np.transpose(v1),
    np.transpose(v2),
    np.transpose(v3),
]
print(f'reciprocal matrix = {matrix}')
inverse_matrix = np.linalg.inv(matrix)
print(f'reciprocal matrix inverse = {np.linalg.inv(matrix)}')

# Reciprocal vectors
r1 = [inverse_matrix[0][0],inverse_matrix[0][1],inverse_matrix[2][-1]]
r2 = [inverse_matrix[1][0],inverse_matrix[1][1],inverse_matrix[1][-1]]
r3 = [inverse_matrix[2][0],inverse_matrix[2][1],inverse_matrix[2][-1]]
print(f'r1 = {r1}, r2 = {r2}, r3 = {r3}')
print('x1 = (r1,X), x2= (r2,X), x3= (r3,X)')
print(f'x1 = {np.dot(r1,vector)}, x2 = {np.dot(r2,vector)}, x3 = {np.dot(r3,vector)}')
print(f"x_expanded = {np.dot(r1,vector)}v1 + {np.dot(r2,vector)}v2 + {np.dot(r3,vector)}v3 ")






