import numpy as np
from matplotlib.pyplot import *

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import os
import sys
from config.config import *
import pandas as pd

if __name__ == '__main__':

	rcParams['text.usetex'] = False



#for num in [9, 18]: 
	for num in [8]:

#	B1_values = [np.logspace(0., 5., num=6), np.logspace(0., 5., num=6), np.logspace(0., 5., num=6), np.logspace(0., 5., num=6)]
#	B2_values = [np.logspace(0., 8., num=num), np.logspace(1., 9., num=num), np.logspace(2., 10., num=num), np.logspace(3., 11., num=num)]
#	B2_values = [np.logspace(0., 2., num=num), np.logspace(2., 4., num=num), np.logspace(4., 6., num=num), np.logspace(6., 8., num=num), np.logspace(8., 10., num=num)]
#	B2_values = [np.logspace(8., 10., num=num), np.logspace(10., 12., num=num), np.logspace(12., 14., num=num)]
#	B2_values = [np.logspace(0., 2., num=num), np.logspace(2., 4., num=num), np.logspace(4., 6., num=num), np.logspace(6., 8., num=num), np.logspace(8., 10., num=num), np.logspace(10., 12., num=num), np.logspace(12., 14., num=num)]
		B2_values = []
		ranges = [(0., 1.), (1., 2.), (2., 3.), (3., 4.), (4., 5.), (5., 6.), (6., 7.), (7., 8.), (8., 9.),(9., 10.), (10., 11.), (11., 12.)]

		for start, end in ranges:
			B2_values.append(np.unique(np.logspace(start, end, num=num)))

#	B1_values = [np.logspace(0., 5., num=6)] * len(B2_values)
		B1_values = [np.logspace(0., 5., num=6)] * len(B2_values)
		for B1, B2 in zip(B1_values, B2_values):

			B2_max, B2_min, B2_length = map(lambda x: x(B2), [np.max, np.min, len])
			B1_max, B1_min, B1_length = map(lambda x: x(B1), [np.max, np.min, len])
			B1_min_exp, B1_max_exp, B2_min_exp, B2_max_exp = map(lambda x: int(np.floor(np.log10(x))), [B1_min, B1_max, B2_min, B2_max])
			reg_name = f"B1_{B1_min_exp}_{B1_max_exp}_{B1_length}_B2_{B2_min_exp}_{B2_max_exp}_{B2_length}"

			# Initialize a list to store the values
			Opinf = []
			Opinf_wefr = []
#		r_list = [6, 12, 18, 24, 30, 36, 42, 48]
#		r_list = [6, 12, 18, 24]
#		r_list = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 30, 36, 42]
			r_list = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
			for r in r_list:
#			fmode_list = np.linspace(-1, max(-r,-10), min(r,10), dtype=int)
#			fmode_list = np.linspace(-1, -5, 5, dtype=int)
#			fmode_list = np.arange(-2, -min(r+1, 21), -2, dtype=int)
#			fmode_list = np.arange(-2, -min(r+1, 21), -2, dtype=int)
#			fmode_list = np.linspace(-1, max(-r,-10), min(r,10), dtype=int)
				fmode_list = np.linspace(-1, max(-r,-5), min(r,5), dtype=int)
				chi_list = np.logspace(-3., 0., 8)
#			chi_list = np.logspace(-3., 0., 8)

				rcut_max, rcut_min, rcut_length = map(lambda x: x(fmode_list), [np.max, np.min, len])
				chi_max, chi_min, chi_length = map(lambda x: x(chi_list), [np.max, np.min, len])
				chi_min_exp, chi_max_exp = map(lambda x: int(np.floor(np.log10(x))), [chi_min, chi_max])
				reg_f_name = f"rcut_{rcut_min}_{rcut_max}_{rcut_length}_chi_{chi_min_exp}_{chi_max_exp}_{chi_length}"

				r2 = int(r)

				# Read Opinf errors
				dir_name = os.path.join(dir_save, reg_name, f'OpInf_r{r}')
				optimal_params_file = dir_name+f"/optimal_params_std_OpInf_with_reg_r{r}_nof.npz"
				optimal_params_nof = np.load(optimal_params_file)

				# Check if optimal_params_nof is empty
				if not optimal_params_nof:
					# Store default values if empty
					Opinf.append({
						'r': r,
						'opt_train_err': 2,
						'opt_val_err': 2,
						'opt_test_err': 2
					})
					label_nof="OpInf with std reg blows up"
				else:
					# Store the desired values in the list
					Opinf.append({
						'r': r,
						'opt_train_err': optimal_params_nof['opt_train_err'],
						'opt_val_err': optimal_params_nof['opt_val_err'],
						'opt_test_err': optimal_params_nof['opt_test_err']
					})
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

				# Read Opinf with efr errors
				dir_name = os.path.join(dir_save, reg_name, reg_f_name, f'OpInf_w_efr_r{r}')
				optimal_params_file = dir_name+f"/optimal_params_std_OpInf_with_reg_r{r}.npz"
				optimal_params = np.load(optimal_params_file)

				if not optimal_params:
					Opinf_wefr.append({
						'r': r,
						'opt_train_err': 2,
						'opt_val_err': 2,
						'opt_test_err': 2
					})
					label_f = (
						"OpInf with std reg + filter blows up\n"
					)
				else:
					Opinf_wefr.append({
						'r': r,
						'opt_train_err': optimal_params['opt_train_err'],
						'opt_val_err': optimal_params['opt_val_err'],
						'opt_test_err': optimal_params['opt_test_err']
					})
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


	
			# Convert Opinf and Opinf_wefr lists to pandas DataFrames
			Opinf= pd.DataFrame(Opinf)
			Opinf_wefr= pd.DataFrame(Opinf_wefr)

			# Example: Extract specific columns
			opt_train_err_Opinf = Opinf['opt_train_err']
			opt_train_err_Opinf_wefr = Opinf_wefr['opt_train_err']

			rcParams['lines.linewidth'] = 0
			rc("figure", dpi=300)           # High-quality figure ("dots-per-inch")
			rc("text", usetex=False )         # Crisp axis ticks
			rc("font", family="serif")      # Crisp axis labels
			rc("legend", edgecolor='none')  # No boxes around legends
			rcParams["figure.figsize"] = (9, 4)
			rcParams.update({'font.size': 12})
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



			rc("figure",facecolor='w')
			rc("axes",facecolor='w',edgecolor='k',labelcolor='k')
			rc("savefig",facecolor='w')
			rc("text",color='k')
			rc("xtick",color='k')
			rc("ytick",color='k')


			ylbs = [r'$\varepsilon_{\mathrm{train}}$',r'$\varepsilon_{\mathrm{val}}$', r'$\varepsilon_{\mathrm{test}}$']
			ls = ['-', '--', ':']
			key = ['opt_train_err', 'opt_val_err', 'opt_test_err']
			ylim = [[0, 1.5], [0, 1.5], [0, 1.5]]

			fig, ax1 = subplots(1, 1, sharex=True)
			for ylb, k, ym, linestyle in zip(ylbs, key, ylim, ls):


				ax1.plot(Opinf['r'], Opinf[k], linestyle=linestyle, marker='o', color=color1, lw=1.25, label='Opinf, ' + ylb)
				ax1.plot(Opinf_wefr['r'], Opinf_wefr[k], linestyle=linestyle, marker='x', color=color2, lw=1.25, label=' Opinf + efr, ' + ylb)
	#		ax1.set_ylabel(ylb)

			ax1.set_xlabel('Reduced-dimension r')
			ax1.set_ylim(ym)
			ax1.set_xticks(r_list)

			ax1.spines['right'].set_visible(False)
			ax1.spines['top'].set_visible(False)
			ax1.yaxis.set_ticks_position('left')
			ax1.xaxis.set_ticks_position('bottom')

	#	lines_labels = [ax1.get_legend_handles_labels()]
	#	lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

	#	legend = ax1.legend(lines, labels, loc=(0.02, -0.75), ncol=2, frameon=False)
			legend = ax1.legend(loc=0, ncol=3, fontsize=10, frameon=False)
	#	colors = [color2, color1]
	#	for j, text in enumerate(legend.get_texts()):
	#		text.set_color(colors[j])

			fig.suptitle(r'{} values of $\beta^1$ in [$10^{{{}}}$, $10^{{{}}}$], {} values of $\beta^2$ in [$10^{{{}}}$, $10^{{{}}}$]'.format(B1_length, B1_min_exp, B1_max_exp, B2_length, B2_min_exp, B2_max_exp))

			tight_layout()

			dir_name = os.path.join(dir_save, reg_name, reg_f_name)
			print(dir_name)
			fig_name = dir_name + f'/errors.png'

			savefig(fig_name, pad_inches=3)
			close()
