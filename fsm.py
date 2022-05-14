import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True)
def gsi(isweepStart, isweepEnd, istep, jsweepStart, jsweepEnd, jstep, u):
    bx = len(u)-1
    by = len(u[0])-1
    for i in range(isweepStart, isweepEnd, istep):
        for j in range(jsweepStart, jsweepEnd, jstep):
            # Local Solver
            if not((i == 0) or (i == bx)):
                a = min(u[i-1][j], u[i+1][j])
            else:
                if((i == 0)):
                    a = min(u[i][j], u[i+1][j])
                else:
                    a = min(u[i][j], u[i-1][j])
            if not((j==0) or (j == by)):
                b = min(u[i][j-1], u[i][j+1])
            else:
                if((j == 0)):
                    b = min(u[i][j], u[i][j+1])
                else:
                    b = min(u[i][j], u[i][j-1])
            ubar = inf
            if(abs(a-b) >= f[i][j]*h):
                ubar = min(a,b) + f[i][j]*h
            else:
                ubar = (a + b + np.sqrt(2*(f[i][j]*h)**2 - (a - b)**2))/2
            u[i][j] = min(ubar, u[i][j])
    return u
def norm_2(mat1, mat2, h):
    return np.sqrt(sum(sum(abs(mat1-mat2) ** 2))* h**2)

    
if __name__ == "__main__":
    gridNum =500
    center = round((gridNum)/2.0)
    inf = 10**6 # Assign Large positive value as Sup-Viscosity Solution
    h = 0.02 # 
    u = np.zeros((gridNum+1, gridNum+1))
    f = np.zeros((gridNum+1, gridNum+1))
    for i in range(0, gridNum+1, 1):
        for j in range(0, gridNum+1, 1):
            u[i][j] = inf
    # ------  A Layer Model ----------------------
    for i in range(0, gridNum+1, 1):
        for j in range(0, gridNum+1, 1):
            if i < center:
                f[i][j] = 1/1
            else:
                f[i][j] = 1/2
    #---------------------------------------------
    src_x = 0
    src_y = 0
    u[src_x][src_y] = 0
    # --------------------------------------------
    dict1 = {
    "1": [gridNum,-1,0, gridNum,-1,0],
    "2": [gridNum,-1,0, 0,1,gridNum+1],
    "3": [0, 1,gridNum+1, gridNum,-1,0],
    "4": [0, 1,gridNum+1, 0,1,gridNum+1]
    }
    MaxIteration = 3 # In simple model error less than tolerance after 1-2 Iteration 
    last_u = u.copy()
    # While(error<tolerance)
    for i in range(MaxIteration):
        [isweepStart,istep,isweepEnd,jsweepStart,jstep, jsweepEnd] = dict1["1"]
        u = gsi(isweepStart, isweepEnd, istep, jsweepStart, jsweepEnd, jstep, u)
        [isweepStart,istep,isweepEnd,jsweepStart,jstep, jsweepEnd] = dict1["2"]
        u = gsi(isweepStart, isweepEnd, istep, jsweepStart, jsweepEnd, jstep, u)    
        [isweepStart,istep,isweepEnd,jsweepStart,jstep, jsweepEnd] = dict1["3"]
        u = gsi(isweepStart, isweepEnd, istep, jsweepStart, jsweepEnd, jstep, u)
        [isweepStart,istep,isweepEnd,jsweepStart,jstep, jsweepEnd] = dict1["4"]
        u = gsi(isweepStart, isweepEnd, istep, jsweepStart, jsweepEnd, jstep, u)
        print(norm_2(u, last_u,h))
        last_u = u.copy()
    print('---------------')
    
    # ------  Show some figure ----------------------
    fig = plt.figure(figsize=(5, 5), dpi=300)  
    ax = plt.gca()  
    plt.imshow(u)
    ax.invert_yaxis()
    plt.title("FSM")
    cbar = plt.colorbar(fraction=0.05, pad=0.05)
    plt.show()
    
    fig = plt.figure(figsize=(5, 5), dpi=300)  
    ax = plt.gca()  
    print(len(u))
    wait_item = abs(u)
    ctr = plt.contour(wait_item, levels=14,rightside_up =True)
    plt.clabel(ctr,fontsize=6,colors=('r'))
    k = plt.imshow(1/f)
    plt.show()

