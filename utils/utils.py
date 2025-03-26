import numpy as np

def compute_Qhat_sq(Qhat):
	"""
	compute_Qhat_sq returns the non-redundant terms in Qhat squared

	:Qhat: reduced data

	:return: Qhat_sq containing the non-redundant in Qhat squared
	"""

	if len(np.shape(Qhat)) == 1:

	    r 		= np.size(Qhat)
	    prods 	= []
	    for i in range(r):
	        temp = Qhat[i]*Qhat[i:]
	        prods.append(temp)

	    Qhat_sq = np.concatenate(tuple(prods))

	elif len(np.shape(Qhat)) == 2:
	    K, r 	= np.shape(Qhat)    
	    prods 	= []
	    
	    for i in range(r):
	        temp = np.transpose(np.broadcast_to(Qhat[:, i], (r - i, K)))*Qhat[:, i:]
	        prods.append(temp)
	    
	    Qhat_sq = np.concatenate(tuple(prods), axis=1)

	else:
	    print('invalid input!')

	return Qhat_sq

def compute_train_err(Qhat_train, Qtilde_train):
	"""
	compute_train_err computes the OpInf training error

	:Qhat_train: 	reference training data
	:Qtilde_train: 	OpInf approximate solution

	:return: train_err containing the value of the training error
	"""
	train_err = np.max(np.sqrt(np.sum( (Qtilde_train - Qhat_train)**2, axis=1) / np.sum(Qhat_train**2, axis=1)))

	return train_err

def solve_opinf_difference_model(qhat0, n_steps_pred, dOpInf_red_model):
	"""
	solve_opinf_difference_model solves the discrete OpInf ROM for n_steps_pred over the target time horizon (training + prediction)

	:qhat0: 			reduced initial condition Qtilde0=np.matmul (Vr.T, q[:, 0]
	:n_steps_pred: 		number of steps over the target time horizon to solve the OpInf reduced model
	:dOpInf_red_model: 	dOpInf ROM

	:return: contains_nan flag indicating NaN presence in in the Qtilde_train reduced solution, Qtilde
	"""

	Qtilde    		= np.zeros((np.size(qhat0), n_steps_pred))
	contains_nans  	= False

	Qtilde[:, 0] = qhat0
	for i in range(n_steps_pred - 1):
	    Qtilde[:, i + 1] = dOpInf_red_model(Qtilde[:, i])

	if np.any(np.isnan(Qtilde)):
	    contains_nans = True

	return contains_nans, Qtilde.T

def solve_opinf_difference_model_wefr(qhat0, n_steps_pred, dOpInf_red_model, fmode, chi):
	"""
	solve_opinf_difference_model solves the discrete OpInf ROM for n_steps_pred over the target time horizon (training + prediction)

	:qhat0: 			reduced initial condition Qtilde0=np.matmul (Vr.T, q[:, 0]
	:n_steps_pred: 		number of steps over the target time horizon to solve the OpInf reduced model
	:dOpInf_red_model: 	dOpInf ROM

	:return: contains_nan flag indicating NaN presence in in the Qtilde_train reduced solution, Qtilde
	"""

	Qtilde    		= np.zeros((np.size(qhat0), n_steps_pred))
	contains_nans  	= False

	Qtilde[:, 0] = qhat0
	for i in range(n_steps_pred - 1):
		Qtmp = dOpInf_red_model(Qtilde[:, i])
		Qfilter = np.copy(Qtmp)
		Qfilter[fmode:] = 0 # ROM projection filter
		Qtilde[:, i + 1] = (1-chi)*Qtmp+ chi*Qfilter# Relax step

	if np.any(np.isnan(Qtilde)):
		contains_nans = True

	return contains_nans, Qtilde.T