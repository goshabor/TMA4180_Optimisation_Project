import bfgs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def BFGS_2(structure, x_0, f, mu, H_0, tol = 10**-8): #Note> p in this code is the direction, not the fixed nodes
    #Start
    x_k = x_0
    H_k = np.eye(3*(structure.N-structure.M))
    grad_E_k = structure.grad_E(x_k, mu)
    potential_energy = []
    iterations = 0
    
    #Run until convergence
    while np.linalg.norm(grad_E_k) > tol and iterations < 1000:
        
        potential_energy.append(structure.E(x_k, mu))
        iterations += 1

        p_k = bfgs.search_direction(H_k, grad_E_k) #Search direction

        a = bfgs.line_search_wolfe(structure.E, structure.grad_E, x_k, p_k, mu) #Line search

        x_kp1 = x_k + a*p_k #Calculating the next x-value

        #Gradient in x-value and gradient change
        grad_E_kp1 = structure.grad_E(x_kp1, mu)
        s_k = x_kp1 - x_k
        y_k = grad_E_kp1 - grad_E_k

        #Updates
        H_kp1 = bfgs.H_update(H_k, s_k, y_k)
        x_k = x_kp1.copy()
        H_k = H_kp1.copy()
        grad_E_k = grad_E_kp1.copy()
        
    print("Number of iterations: ", iterations, "\nEucledian norm of gradient: ", np.linalg.norm(grad_E_k))
    return x_k, grad_E_k, potential_energy

def f(x1, x2):
    #Floor function
    return (x1**2 + x2**2) / 20

def plot_3d_points_2(points, cable_structure, bar_structure = [], title1='3D Plot of Points with Lines', title2='2D View'):
    # Extracting x, y, z coordinates from points array
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Create a new figure for the plots
    fig = plt.figure(figsize=(14, 7))  # Increased figure size for better visibility

    # 3D plot on the left
    ax1 = fig.add_subplot(121, projection='3d')  # 1 row, 2 cols, subplot 1
    ax1.scatter(x, y, z, c='r', marker='o')

    # Plotting the floor using function f
    X1, X2 = np.meshgrid(np.linspace(min(x), max(x), 50), np.linspace(min(y), max(y), 50))
    Z = f(X1, X2)
    ax1.plot_surface(X1, X2, Z, alpha=0.5, rstride=100, cstride=100, color='gray', edgecolors='k')  # Semi-transparent floor

    # Define colors for the points
    colors = ['r'] * len(x)
    colors[:4] = ['r', 'r', 'r', 'r']
    ax1.scatter(x, y, z, c=colors, marker='o')

    # Loop through the matrix to add lines for the 3D plot
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            if cable_structure[i, j] != 0:
                ax1.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 'k--')  # Dashed lines for cables
            if bar_structure[i, j] != 0:
                ax1.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 'k-')  # Solid lines for bars

    # Set labels for the 3D axes
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.set_zlabel('Z axis')
    ax1.set_title(title1)

    # 2D plot (top-down view) on the right
    ax2 = fig.add_subplot(122)  # 1 row, 2 cols, subplot 2
    ax2.scatter(x, y, c='r', marker='o')

    # Loop through the matrix to add lines for the 2D plot
    for i in range(n):
        for j in range(i+1, n):
            if cable_structure[i, j] != 0:
                ax2.plot([x[i], x[j]], [y[i], y[j]], 'k--')  # Dashed lines for cables
            if bar_structure[i, j] != 0:
                ax2.plot([x[i], x[j]], [y[i], y[j]], 'k-')  # Solid lines for bars
    
        ax2.text(x[i] + 0.1, y[i] + 0.1, str(i + 1), color='blue', fontsize=9)

    # Set labels for the 2D axes
    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    ax2.set_title(title2)

    # Show the plot
    plt.show()