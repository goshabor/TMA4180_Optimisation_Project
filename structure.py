import numpy as np

class Structure:
    def __init__(self, N, k, cable_instructions, p, x, mg, bar_instructions=[], pg=0, c=0, f=None):
        self.N = N #Constant
        self.M = int(len(p)/3) #Number of free variables
        self.k = k # Constant
        self.cable_grid = self.initialize_cable_grid(cable_instructions) #Connection matrix and connection values
        self.bar_grid = self.initialize_bar_grid(bar_instructions)
        self.p = p #Fixed nodes
        self.mg = mg #N-M last
        self.pg = pg #rho*g 
        self.x = np.concatenate([p, x]) #Fixed + variable nodes
        self.x_only = x #All variable nodes
        self.c = c #Constant
        self.f = f #Floor function
        
    def initialize_cable_grid(self, cable_instructions):
        #N x N zeros array
        target_array = [[0 for _ in range(self.N)] for _ in range(self.N)]

        for instruction in cable_instructions:
            #Change indice base and insert values to zeros array
            row = instruction[0] - 1  
            col = instruction[1] - 1  
            value = instruction[2]
            target_array[row][col] = value

        return np.array(target_array)
    
    def initialize_bar_grid(self, bar_instructions):
        #N x N zeros array
        target_array = [[0 for _ in range(self.N)] for _ in range(self.N)]

        for instruction in bar_instructions:
             #Change indice base and insert values to zeros array
            row = instruction[0] - 1  
            col = instruction[1] - 1  
            value = instruction[2]
            target_array[row][col] = value

        return np.array(target_array) 

    def change_x(self, x_new):
        #Changes x to new inputed value
        self.x[3*self.M:] = x_new
        self.x_only = x_new
    
    def E_ext(self, x=None): 
        #Potential energy from external forces, calculated only for variating nodes, mg*z 
        if x is None:
            x = self.x_only
        x = np.concatenate([self.p, x])
        return np.sum(self.mg*x[2::3])
    
    def E_cable_elast(self, x=None): 
        #Cable elastic potential energy, calculated for all strings between nodes
        if x is None:
            x = self.x_only
        E_sum = 0
        x = np.concatenate([self.p, x])
        for i, row in enumerate(self.cable_grid):  #Correctly get the index and row
            for j, l_ij in enumerate(row):  #Correctly get the index and column value
                if l_ij:  #If l_ij (which is self.l[i][j]) not zero
                    distance = np.linalg.norm(x[i*3:(i+1)*3] - x[j*3:(j+1)*3])
                    if distance > l_ij:
                        E_sum += self.k/(2*l_ij**2) * (distance - l_ij)**2
        return E_sum
        
    def E_bar(self, x=None):       
        #Bar potential energy, calculated for all strings between nodes
        if x is None:
            x = self.x_only
        E_sum = 0
        x = np.concatenate([self.p, x])
        for i, row in enumerate(self.bar_grid):  #Correctly get the index and row
            for j, l_ij in enumerate(row):  #Correctly get the index and column value
                if l_ij:  #If l_ij (which is self.l[i][j]) not 0
                    distance = np.linalg.norm(x[i*3:(i+1)*3] - x[j*3:(j+1)*3])
                    E_sum += self.c/(2*l_ij**2) * (distance - l_ij)**2 #E-bar elast
                    E_sum += self.pg*l_ij/2 * (x[i*3+2] + x[j*3+2])  #E-bar grav
                    
        return E_sum
    
    def quadratic_penalty(self, x=None, mu=0):
        #Penalty if inequality is not fullfilled
        E = 0
        if x is None:
            x = self.x_only
        x = np.concatenate([self.p, x])

        #Check
        for i in range(self.N-self.M):
            if self.f(x[0::3][i],x[1::3][i]) > x[2::3][i]:
                E += mu/2 * (self.f(x[0::3][i],x[1::3][i]) - x[2::3][i])**2 / 2 #Penalty
                
        return E
    
    def grad_E(self, x=None, mu=0):
        #Function for energy gradient calculation
        if x is None:
            x = self.x_only
        x = np.concatenate([self.p, x])
        
        #Gradient vector
        gradient = np.zeros(3*(self.N-self.M))

        #For cable
        for i in range(self.N-self.M): 
            grad_i = np.zeros(3)
            grad_i[2] = self.mg
            #Checking if there are cables between nodes + gradient
            for j, l_ij in enumerate(self.cable_grid[self.M + i]):
                if l_ij: #Check if there are cables
                    distance = np.linalg.norm(x[(self.M+i)*3:(self.M+i+1)*3] - x[j*3:(j+1)*3]) 
                    if distance > l_ij:
                        grad_i += self.k/l_ij**2 * (distance - l_ij) / distance * (x[(self.M+i)*3:(self.M+i+1)*3] - x[j*3:(j+1)*3]) #Gradient calculation
            #Checking if there are cables between nodes + gradient
            for j, l_ji in enumerate([row[self.M + i] for row in self.cable_grid]):
                if l_ji: #Check if there are cables 
                    distance = np.linalg.norm(x[j*3:(j+1)*3] - x[(self.M+i)*3:(self.M+i+1)*3])
                    if distance > l_ji:
                        grad_i += self.k/(l_ji**2) * (1 - l_ji/distance) *(x[(self.M+i)*3:(self.M+i+1)*3] - x[j*3:(j+1)*3]) #Adds element to gradient
            gradient[3*i:3*(i+1)] += grad_i #Goes to gradient vector

        #For bar
        for i in range(self.N-self.M): 
            grad_i = np.zeros(3)
            #Checking if there are bars between nodes + gradient
            for j, l_ij in enumerate(self.bar_grid[self.M + i]):
                if l_ij: #Check for bars
                    distance = np.linalg.norm(x[(self.M+i)*3:(self.M+i+1)*3] - x[j*3:(j+1)*3])
                    #Add to gradient
                    grad_i += self.c/l_ij**2 * (distance - l_ij) / distance * (x[(self.M+i)*3:(self.M+i+1)*3] - x[j*3:(j+1)*3])
                    grad_i[2] += self.pg*l_ij/2

            #Checking if there are bars between nodes + gradient        
            for j, l_ji in enumerate([row[self.M + i] for row in self.bar_grid]):
                if l_ji: #Check bars and calculate gradient
                    distance = np.linalg.norm(x[j*3:(j+1)*3] - x[(self.M+i)*3:(self.M+i+1)*3])
                    grad_i += self.c/(l_ji**2) * (1 - l_ji/distance) *(x[(self.M+i)*3:(self.M+i+1)*3] - x[j*3:(j+1)*3]) 
                    grad_i[2] += self.pg*l_ji/2
            
            gradient[3*i:3*(i+1)] += grad_i #Add to gradient vector
            
        #Quadratic Penalty
        if self.f is not None:
            for i in range(self.N-self.M):
                if self.f(x[0::3][i],x[1::3][i]) > x[2::3][i]:
                    gradient[3*i] -= mu/10 * x[0::3][i] * (x[2::3][i] - self.f(x[0::3][i],x[1::3][i]))
                    gradient[3*i+1] -= mu/10 * x[1::3][i] * (x[2::3][i] - self.f(x[0::3][i],x[1::3][i]))
                    gradient[3*i+2] += mu * (x[2::3][i] - self.f(x[0::3][i],x[1::3][i]))
                    
        
        return gradient
        
    def E(self, x=None, mu=0):
        #Structure energy calculation
        if x is None:
            x = self.x_only
        if self.f is None: #Without floor function
            return self.E_ext(x) + self.E_cable_elast(x) + self.E_bar(x)
        return self.E_ext(x) + self.E_cable_elast(x) + self.E_bar(x) + mu*self.quadratic_penalty(x) #With penalty
        