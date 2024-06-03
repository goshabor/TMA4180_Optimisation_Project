import structure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def search_direction(H, grad_E):
    #Finds search direction
    return  - np.dot(H, grad_E)

def H_update(H_k, s_k, y_k):  #Hessian update
    #Ensure s_k and y_k are column vectors
    s_k = s_k.reshape(-1, 1)
    y_k = y_k.reshape(-1, 1)

    
    rho_k = 1.0 / np.dot(y_k.T, s_k) 
    if np.abs(rho_k) < 10**(-15): #Check stability
        print("not good")
    I = np.eye(len(s_k))
    
    #Update 
    term1 = I - rho_k * np.dot(s_k, y_k.T)
    term2 = I - rho_k * np.dot(y_k, s_k.T)
    term3 = rho_k * np.dot(s_k, s_k.T)
    
    #Updated Hessian
    H_kp1 = np.dot(np.dot(term1, H_k), term2) + term3
    
    return H_kp1

def line_search_wolfe(f, grad_f, x_k, p_k, mu=0, rho=2, c1=0.01, c2=0.9, alpha_max=1.0):
    #Line search with Wolfe conditions
    alpha_low = 0
    alpha_high = alpha_max/10
    descent = np.dot(grad_f(x_k, mu), p_k)
    fxk = f(x_k)

    #Extrapolation
    while True:
        alpha = alpha_high
        if not (f(x_k + alpha * p_k) <= fxk + c1 * alpha * descent) or \
           (np.dot(grad_f(x_k + alpha * p_k, mu), p_k) >= c2 * descent):
            break
        alpha_low = alpha_high
        alpha_high *= rho  #Increase step size
        
        if alpha_high > alpha_max:  #Avoid exceeding max step size
            alpha_high = alpha_max
            break

    #Interpolation
    for _ in range(15):  #Iter limit for safety
        alpha = (alpha_low + alpha_high) / 2
        if f(x_k + alpha * p_k) <= fxk + c1 * alpha * descent and \
           np.dot(grad_f(x_k + alpha * p_k, mu), p_k) >= c2 * descent:
            break  #Good enough alpha
        elif np.dot(grad_f(x_k + alpha * p_k, mu), p_k) < c2 * descent:
            alpha_low = alpha  #Move up the lower bound
        else:
            alpha_high = alpha  #Reduce upper bound if Armijo condition fails
    return alpha

def BFGS(structure, x_0,tol = 0.01, mu=0): #Note: p is the search direction, not the fixed nodes
    #Start
    x_k = x_0
    H_k = np.eye(3*(structure.N-structure.M))
    grad_E_k = structure.grad_E(x_k, mu)
    iterations = 0
    
    #Until convergence
    while np.linalg.norm(grad_E_k) > tol:
        
        iterations += 1

        p_k = search_direction(H_k, grad_E_k) #Search direction

        a = line_search_wolfe(structure.E, structure.grad_E, x_k, p_k, mu) #Line search

        x_kp1 = x_k + a*p_k #Calculating the next x-value

        #New gradient and gradient change
        grad_E_kp1 = structure.grad_E(x_kp1, mu)
        s_k = x_kp1 - x_k
        y_k = grad_E_kp1 - grad_E_k

        #Updates
        H_kp1 = H_update(H_k, s_k, y_k)
        x_k = x_kp1.copy()
        H_k = H_kp1.copy()
        grad_E_k = grad_E_kp1.copy()
    
    print("Number of iterations: ", iterations, "\nEucledian norm of gradient: ", np.linalg.norm(grad_E_k))
    return x_k

def plot_3d_points(points, cable_structure, bar_structure = [], title='3D Plot of Points with Lines'): 
    # Extracting x, y, z coordinates from points array
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Create a new figure for the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for the points
    ax.scatter(x, y, z, c='r', marker='o')
    
    colors = ['r'] * len(x)
    colors[:4] = ['k', 'k', 'k', 'k'] 

    ax.scatter(x, y, z, c=colors, marker='o')

    # Loop through the matrix to add lines where matrix[i, j] is non-zero
    n = len(points)  # Assuming matrix is NxN and matches the number of points
    for i in range(n):
        for j in range(i+1, n):  # Only go through upper triangle to avoid double-drawing lines
            if cable_structure[i, j] != 0:  # Check if there's a connection between points i and j
                # Draw a line between points i and j
                ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 'k--')  # Using blue lines for connections
                
    for i in range(n):
        for j in range(i+1, n):  # Only go through upper triangle to avoid double-drawing lines
            if bar_structure[i, j] != 0:  # Check if there's a connection between points i and j
                # Draw a line between points i and j
                ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 'k-')  # Using green lines for connections

    # Set labels for the 3D axes
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Title
    ax.set_title(title)

    # Show the plot
    plt.show()