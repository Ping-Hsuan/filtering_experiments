import numpy as np

def get_x_sq(X):
    if len(np.shape(X))==1: # if X is a vector
        r = np.size(X)
        prods = []
        for i in range(r):
            temp = X[i]*X[i:]
            prods.append(temp)
        X2 = np.concatenate(tuple(prods))

    elif len(np.shape(X))==2: # if X is a matrix
        K,r = np.shape(X)
        
        prods = []
        for i in range(r):
            temp = np.transpose(np.broadcast_to(X[:,i],(r-i,K)))*X[:,i:]
            prods.append(temp)
        X2 = np.concatenate(tuple(prods),axis=1)

    else:
        print('invalid input size for helpers.get_x_sq')
    return X2

def solve_opinf_difference_model(s0, n_steps, f):

    s       = np.zeros((np.size(s0), n_steps))
    is_nan  = False

    s[:, 0] = s0
    for i in range(n_steps - 1):

        s[:, i + 1] = f(s[:, i])

        if np.any(np.isnan(s[:, i + 1])):
            print('NaN encountered at iteration '+str(i + 1))

            is_nan = True

            break

    return is_nan, s

r               = 22
n_steps         = 441
training_end    = 294


red_operators = np.load('OpInf_results/red_operators_std_OpInf_mild_reg_r' + str(r) + '.npz')

A = red_operators['lin']
F = red_operators['quad']
C = red_operators['const']

f = lambda x: A @ x + F @ get_x_sq(x) + C


eigvals, eigvecs = np.linalg.eig(A)

print(eigvals)

u0                  = np.load('OpInf_results/red_init_cond_r' + str(r) + '.npy')
is_nan, Xhat_OpInf  = solve_opinf_difference_model(u0, n_steps, f)

print(Xhat_OpInf)