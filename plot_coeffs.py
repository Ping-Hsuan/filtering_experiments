import numpy as np
from matplotlib.pyplot import *

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import os
import sys
from config.config import *

if __name__ == '__main__':

	rcParams['text.usetex'] = False
	if len(sys.argv) > 1:
		r = sys.argv[1]
	else:
		r = 'noefr'

	r1 = 22
	r2 = int(r)

	# dt = 1e-4
	dt = 1e-4

	ti = 0.3805 #0.00099338
	tf0 = ti+dt*1267 #0.00137569
	tf = ti+dt*2535 #0.00137569
	tff = ti+dt*3750

	plotting_end = 3751 #2536
	training_end = 1268
	val_end = 2536

	t  = np.arange(ti, tf, dt)
	t_all  = np.arange(ti, tff, dt)
	print(t_all.shape)
	t_OpInf = t

	dir_name = os.path.join(dir_save, reg_name, f'OpInf_r{r}')
	OpInf_file_nof = lambda x: dir_name+'/red_sol_std_OpInf_with_reg_r' + str(x) + '_nof.npy'
	optimal_params_file = dir_name+f"/optimal_params_std_OpInf_with_reg_r{r}_nof.npz"
	optimal_params_nof = np.load(optimal_params_file)

	if not optimal_params_nof:
		print("The npz file is empty.")
		idx = 5#optimal_params_nof['idx']-8
		label_nof="OpInf with std reg blows up"
	else:
		idx = plotting_end

		label_nof = (
			"OpInf with std reg \n"
			+ r"($\varepsilon_{{\mathrm{{train}}}}$ = {:.3f}, "
			r"$\varepsilon_{{\mathrm{{val}}}}$ = {:.3f}, "
			r"$\varepsilon_{{\mathrm{{test}}}}$ = {:.3f})".format(
				optimal_params_nof['opt_train_err'],
				optimal_params_nof['opt_val_err'],
				optimal_params_nof['opt_test_err']
			)
		)

	OpInf_red_coeff_nof = np.load(OpInf_file_nof(r2))
	print(OpInf_red_coeff_nof.shape)
#try:
#	OpInf_red_coeff_nof = np.load(OpInf_file_nof(r2))
#	if OpInf_red_coeff_nof is None:
#		raise ValueError("Loaded object is None")
#except Exception as e:
#	print(f"Error loading OpInf_file_nof: {e}")
#	OpInf_red_coeff_nof = np.array([])  # or handle it a
    
	dir_name = os.path.join(dir_save, reg_name, reg_f_name, f'OpInf_w_efr_r{r}')
	OpInf_file_filter = lambda x: dir_name+'/red_sol_std_OpInf_with_reg_r' + str(x) + '.npy'
	OpInf_red_coeff_filter = np.load(OpInf_file_filter(r2))

	optimal_params_file = dir_name+f"/optimal_params_std_OpInf_with_reg_r{r}.npz"
	optimal_params = np.load(optimal_params_file)
	print(optimal_params)

	if not optimal_params:
		print("The npz file is empty.")
		idx_reg = 5#optimal_params_nof['idx']-8
		label_f = "OpInf with std reg + filter blows up"
	else:
		idx_reg = plotting_end
		label_f = (
			"OpInf with std reg + filter\n"
			+ r"($\varepsilon_{{\mathrm{{train}}}}$ = {:.3f}, "
			r"$\varepsilon_{{\mathrm{{val}}}}$ = {:.3f}, "
			r"$\varepsilon_{{\mathrm{{test}}}}$ = {:.3f}, "
			r"$r_{{\mathrm{{cutoff}}}}$ = {}, $\chi_{{\mathrm{{opt}}}}$ = {:.3f})".format(
				optimal_params['opt_train_err'],
				optimal_params['opt_val_err'],
				optimal_params['opt_test_err'],
				optimal_params['fmode_opt'],
				optimal_params['chi_opt']
			)
		)

	ref_data  	  = np.load(Qhat_full_file)
	ref_coeff_all = ref_data

	
	rcParams['lines.linewidth'] = 0
	rc("figure", dpi=300)           # High-quality figure ("dots-per-inch")
	rc("text", usetex=False )         # Crisp axis ticks
	rc("font", family="serif")      # Crisp axis labels
	rc("legend", edgecolor='none')  # No boxes around legends
	rcParams["figure.figsize"] = (9, 5)
	rcParams.update({'font.size': 9})
	# rcParams['figure.conspreded_layout.use'] = True

	charcoal    = [0.1, 0.1, 0.1]
	color1      = '#D55E00'
	color2      = '#0072B2'
	color3      = '#4daf4a'
	
	linestyle1 = '--'
	linestyle2 = (0, (5, 1))
	linestyle3 = '-'
	# linestyle4 = (0, (3, 1, 1, 1))
	linestyle4 = '-'
	linestyle5 = (0, (5, 5))
	

	# rcParams["figure.figsize"] = (6, 6)


	fig1 	= figure()
	ax1 	= fig1.add_subplot(211)
	ax2 	= fig1.add_subplot(212)

	rc("figure",facecolor='w')
	rc("axes",facecolor='w',edgecolor='k',labelcolor='k')
	rc("savefig",facecolor='w')
	rc("text",color='k')
	rc("xtick",color='k')
	rc("ytick",color='k')

	ax1.spines['right'].set_visible(False)
	ax1.spines['top'].set_visible(False)
	ax1.yaxis.set_ticks_position('left')
	ax1.xaxis.set_ticks_position('bottom')

	ax2.spines['right'].set_visible(False)
	ax2.spines['top'].set_visible(False)
	ax2.yaxis.set_ticks_position('left')
	ax2.xaxis.set_ticks_position('bottom')


	pos1 = 0
	pos2 = r2-1

	for ax, pos in zip([ax1, ax2], [pos1, pos2]):
		ax.plot(t_all[:], ref_coeff_all[:, pos], linestyle='-', color=charcoal, lw=1.25, label='ref data')
		ax.plot(t_all[:idx_reg], OpInf_red_coeff_filter[:idx_reg, pos], linestyle='-.', color=color2, lw=1.25, label=label_f)
		ax.plot(t_all[:idx], OpInf_red_coeff_nof[:idx, pos], linestyle='--', color=color1, lw=1.25, label=label_nof)
		ax.set_xlabel('time [sec]')
		ax.set_ylabel('Coeff for POD mode = {}'.format(pos + 1))
		ax.axvline(t_all[training_end - 1], lw=1.25, linestyle='--', color=charcoal)
		ax.axvline(t_all[val_end - 1], lw=1.25, linestyle='--', color=charcoal)

		x_pos_all 	= np.array([0.3805, tf0, tf, round(tff, 4)])
		labels 		= np.array([0.3805, tf0, tf, round(tff, 4)])

		ax.set_xticks(x_pos_all)
		ax.set_xticklabels(labels)
	
	lines_labels = [ax1.get_legend_handles_labels()]
	lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

	legend = ax2.legend(lines, labels, loc = (0.02, -0.75), ncol=2, frameon=False)
	colors = [charcoal, color2, color1]
	for i, text in enumerate(legend.get_texts()):
		text.set_color(colors[i])
	#fig1.suptitle('Regularization parameter space [B2_min = {:.2e}, B2_max = {:.2e}]'.format(B2_min, B2_max))

	fig1.suptitle(r'r = {}, {} values of $\beta^1$ in [$10^{{{}}}$, $10^{{{}}}$], {} values of $\beta^2$ in [$10^{{{}}}$, $10^{{{}}}$]'.format(r2, B1_length, B1_min_exp, B1_max_exp, B2_length, B2_min_exp, B2_max_exp))

	tight_layout()

	fig_name = dir_name +'/coefficients_with_efr.png'

	savefig(fig_name, pad_inches=3)
	close()