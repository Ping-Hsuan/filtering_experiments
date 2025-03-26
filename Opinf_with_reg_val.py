from config.config import *
from utils.utils import *
import os
import sys
import pandas as pd

# Get the input case from command line arguments
if len(sys.argv) > 1:
    case = sys.argv[1]
else:
    case = 'noefr'

print(case)

# load training data
Qhat_full           = np.load(Qhat_full_file)
Qhat_training       = Qhat_full[:nt, :r]

Qhat_full_domain           = Qhat_full

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
opt_val_err             = 1e20


results_list = []

if case == 'efr':

    output_dir = os.path.join(dir_save, reg_name, reg_f_name, f'OpInf_w_efr_r{r}')
    os.makedirs(output_dir, exist_ok=True)

    # loop over all regularization pairs
    for fmode in fmode_list:
        for chi in chi_list:
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
                    contains_nans, Qtilde_OpInf = solve_opinf_difference_model_wefr(qhat0, nt_p, dOpInf_red_model, fmode, chi)

                    # for each candidate regulairzation pair, we compute the training error 
                    # we also save the corresponding reduced solution, learning time and ROM evaluation time
                    # and compute the ratio of maximum coefficient growth in the trial period to that in the training period
                    if contains_nans == False:
                        train_err               = compute_train_err(Qhat_training, Qtilde_OpInf[:nt, :])
                        val_err                 = compute_train_err(Qhat_full_domain[nt:nt_p, :r], Qtilde_OpInf[nt:nt_p, :])
                        max_diff_Qhat_trial     = np.max(np.abs(Qtilde_OpInf - mean_Qhat_train), axis=0)            
                        max_growth_trial        = np.max(max_diff_Qhat_trial)/np.max(max_diff_Qhat_train)


                        if max_growth_trial < max_growth:
                            # Append the current results to a list
                            results_list.append({
                                'train_err': train_err,
                                'val_err': val_err,
                                'beta1': beta1,
                                'beta2': beta2,
                                'fmode': fmode,
                                'chi': chi
                            })


                            if val_err < opt_val_err:
                                print(train_err, beta1, beta2, fmode, chi, val_err)
                                opt_train_err               = train_err
                                opt_val_err                 = val_err
                                Qtilde_OpInf_opt            = Qtilde_OpInf
                                
                                beta1_opt = beta1
                                beta2_opt = beta2
                                fmode_opt = fmode
                                chi_opt = chi
                                Ahat_opt = Ohat[:, :r]
                                Fhat_opt = Ohat[:, r:r+s]
                                chat_opt = Ohat[:, r+s]


    if 'Ahat_opt' not in locals():
        print("Ahat_opt is not defined.")
        print("Skipping simulation as Ahat_opt is None.")

        np.save(os.path.join(output_dir, f'red_sol_std_OpInf_with_reg_r{r}.npy'), Qtilde_OpInf)
        params_filename = output_dir+'/optimal_params_std_OpInf_with_reg_r' + str(r) + '.npz'
        np.savez(params_filename) 
    else:
        # simulate the ROM for the entire interval
        dOpInf_red_model    = lambda x: Ahat_opt @ x + Fhat_opt @ compute_Qhat_sq(x) + chat_opt
        # extract the reduced initial condition from Qhat_1
        qhat0               = Qhat_1[0, :]
            
        # compute the reduced solution over the trial time horizon, which here is the same as the target time horizon
        contains_nans, Qtilde_OpInf = solve_opinf_difference_model_wefr(qhat0, nt_all, dOpInf_red_model, fmode_opt, chi_opt)
        test_err  = compute_train_err(Qhat_full_domain[nt_p:, :r], Qtilde_OpInf[nt_p:, :])

        # save results to disk
        # np.save('./OpInf_results/red_sol_std_OpInf_with_reg_r22.npy', Qtilde_OpInf_opt)

        results_file = os.path.join(output_dir, f'results_r{r}_efr.csv')
        pd.DataFrame(results_list).to_csv(results_file, index=False)

        np.save(os.path.join(output_dir, f'red_sol_std_OpInf_with_reg_r{r}.npy'), Qtilde_OpInf)

        # Save the operators in the same format
        output_filename = output_dir+'/red_operators_std_OpInf_with_reg_r' + str(r) + '.npz'
        np.savez(output_filename, lin=Ahat_opt, quad=Fhat_opt, const=chat_opt)

        # Save the optimal parameters to a file
        params_filename = output_dir+'/optimal_params_std_OpInf_with_reg_r' + str(r) + '.npz'
        np.savez(params_filename, 
                beta1_opt=beta1_opt, 
                beta2_opt=beta2_opt, 
                fmode_opt=fmode_opt, 
                chi_opt=chi_opt, 
                opt_train_err=opt_train_err, 
                B1=B1, 
                B2=B2, 
                r_cut=fmode_list,
                chi=chi_list, 
                opt_val_err=opt_val_err,
                opt_test_err=test_err)

else:

    output_dir = os.path.join(dir_save, reg_name, f'OpInf_r{r}')
    os.makedirs(output_dir, exist_ok=True)

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
                val_err                 = compute_train_err(Qhat_full_domain[nt:nt_p, :r], Qtilde_OpInf[nt:nt_p, :])
                max_diff_Qhat_trial     = np.max(np.abs(Qtilde_OpInf - mean_Qhat_train), axis=0)            
                max_growth_trial        = np.max(max_diff_Qhat_trial)/np.max(max_diff_Qhat_train)

                if max_growth_trial < max_growth:

                    results_list.append({
                        'train_err': train_err,
                        'val_err': val_err,
                        'beta1': beta1,
                        'beta2': beta2,
                    })

                    if val_err < opt_val_err:
                        print(train_err, beta1, beta2, val_err)
                        opt_train_err               = train_err
                        opt_val_err                 = val_err
                        Qtilde_OpInf_opt            = Qtilde_OpInf
                        
                        beta1_opt = beta1
                        beta2_opt = beta2
                        Ahat_opt = Ohat[:, :r]
                        Fhat_opt = Ohat[:, r:r+s]
                        chat_opt = Ohat[:, r+s]

    results_file = os.path.join(output_dir, f'results_r{r}.csv')
    pd.DataFrame(results_list).to_csv(results_file, index=False)

    if 'Ahat_opt' not in locals():
        print("Ahat_opt is not defined.")
        print("Skipping simulation as Ahat_opt is None.")

        # save results to disk
        # np.save('./OpInf_results/red_sol_std_OpInf_with_reg_r22.npy', Qtilde_OpInf_opt)
        np.save(os.path.join(output_dir, f'red_sol_std_OpInf_with_reg_r{r}_nof.npy'), Qtilde_OpInf)
        # Save the optimal parameters to a file
        params_filename = output_dir+'/optimal_params_std_OpInf_with_reg_r' + str(r) + '_nof.npz'
        np.savez(params_filename)
    else:

        # simulate the ROM for the entire interval
        dOpInf_red_model    = lambda x: Ahat_opt @ x + Fhat_opt @ compute_Qhat_sq(x) + chat_opt
        # extract the reduced initial condition from Qhat_1
        qhat0               = Qhat_1[0, :]
            
        # compute the reduced solution over the trial time horizon, which here is the same as the target time horizon
        contains_nans, Qtilde_OpInf = solve_opinf_difference_model(qhat0, nt_all, dOpInf_red_model)
        test_err  = compute_train_err(Qhat_full_domain[nt_p:, :r], Qtilde_OpInf[nt_p:, :])

        # save results to disk
        # np.save('./OpInf_results/red_sol_std_OpInf_with_reg_r22.npy', Qtilde_OpInf_opt)
        np.save(os.path.join(output_dir, f'red_sol_std_OpInf_with_reg_r{r}_nof.npy'), Qtilde_OpInf)
        # Save the optimal parameters to a file
        params_filename = output_dir+'/optimal_params_std_OpInf_with_reg_r' + str(r) + '_nof.npz'
        np.savez(params_filename,
                beta1_opt=beta1_opt,
                beta2_opt=beta2_opt,
                opt_train_err=opt_train_err,
                B1=B1,
                B2=B2,
                opt_val_err=opt_val_err,
                opt_test_err=test_err)