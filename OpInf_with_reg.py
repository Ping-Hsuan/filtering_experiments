from config.config import *
from utils.utils import *

# load training data
Qhat_full           = np.load(Qhat_full_file)
Qhat_training       = Qhat_full[:, :r]

# extract left and right shifted reduced data matrices for the discrete OpInf learning problem
Qhat_1 = Qhat_training[:-1, :]
Qhat_2 = Qhat_training[1:, :]

# column dimension of the data matrix Dhat used in the discrete OpInf learning problem
s = int(r*(r + 1)/2)
d = r + s + 1

# compute the non-redundant quadratic terms of Qhat_1 squared
Qhat_1_sq = compute_Qhat_sq(Qhat_1)

# define the constant part (due to mean shifting) in the discrete OpInf learning problem
K       = Qhat_1.shape[0]
Ehat    = np.ones((K, 1))

# assemble the data matrix Dhat for the discrete OpInf learning problem
Dhat   = np.concatenate((Qhat_1, Qhat_1_sq, Ehat), axis=1)
# compute Dhat.T @ Dhat for the normal equations to solve the OpInf least squares minimization
Dhat_2 = Dhat.T @ Dhat

# compute the temporal mean and maximum deviation of the reduced training data
mean_Qhat_train         = np.mean(Qhat_training, axis=0)
max_diff_Qhat_train     = np.max(np.abs(Qhat_training - mean_Qhat_train), axis=0)
# training error corresponding to the optimal regularization hyperparameters
opt_train_err           = 1e20

# loop over all regularization pairs
for beta1 in B1:
    for beta2 in B2:

        # regularize the linear and constant reduced operators using beta1, and the quadratic operator using beta2
        regg            = np.zeros(d)
        regg[:r]        = beta1
        regg[r : r + s] = beta2
        regg[r + s:]    = beta1
        regularizer     = np.diag(regg)
        Dhat_2_reg      = Dhat_2 + regularizer

        # solve the OpInf learning problem by solving the regularized normal equations
        Ohat = np.linalg.solve(Dhat_2_reg, np.dot(Dhat.T, Qhat_2)).T

        # extract the linear, quadratic, and constant reduced model operators
        Ahat = Ohat[:, :r]
        Fhat = Ohat[:, r:r + s]
        chat = Ohat[:, r + s]

        # define the OpInf reduced model 
        dOpInf_red_model    = lambda x: Ahat @ x + Fhat @ compute_Qhat_sq(x) + chat
        # extract the reduced initial condition from Qhat_1
        qhat0               = Qhat_1[0, :]
        
        # compute the reduced solution over the trial time horizon, which here is the same as the target time horizon
        contains_nans, Qtilde_OpInf = solve_opinf_difference_model(qhat0, nt_p, dOpInf_red_model)

        # for each candidate regulairzation pair, we compute the training error 
        # we also save the corresponding reduced solution, learning time and ROM evaluation time
        # and compute the ratio of maximum coefficient growth in the trial period to that in the training period
        if contains_nans == False:
            train_err               = compute_train_err(Qhat_training, Qtilde_OpInf[:nt, :])
            max_diff_Qhat_trial     = np.max(np.abs(Qtilde_OpInf - mean_Qhat_train), axis=0)            
            max_growth_trial        = np.max(max_diff_Qhat_trial)/np.max(max_diff_Qhat_train)

            if max_growth_trial < max_growth:

                if train_err < opt_train_err:
                    opt_train_err               = train_err
                    Qtilde_OpInf_opt            = Qtilde_OpInf
                    
                    beta1_opt = beta1
                    beta2_opt = beta2

# save results to disk
# np.save('./OpInf_results/red_sol_std_OpInf_with_reg_r22.npy', Qtilde_OpInf_opt)