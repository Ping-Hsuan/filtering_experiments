import numpy as np
from matplotlib.pyplot import *

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


if __name__ == '__main__':

	r1 = 22
	r2 = 22

	ti = 0.00099338
	tf = 0.00137569

	plotting_end = 441
	training_end = 294

	cutoff = 200

	dt = (tf - ti)/441
	t  = np.arange(ti, tf, dt)
	t_OpInf = t


	ref_data 				= lambda x: np.load('OpInf_results/ref_red_sol_training_r' + str(x) + '.npy')
	ref_coeff_training 		= ref_data(r1)

	OpInf_file_with_reg 		= lambda x: 'OpInf_results/red_sol_std_OpInf_with_reg_r' + str(x) + '.npy'
	OpInf_red_coeff_with_reg 	= np.load(OpInf_file_with_reg(r1))

	OpInf_file_mild_reg 		= lambda x: 'OpInf_results/red_sol_std_OpInf_mild_reg_r' + str(x) + '.npy'
	OpInf_red_coeff_mild_reg 	= np.load(OpInf_file_mild_reg(r2))

	print(ref_coeff_training.shape)
	print(OpInf_red_coeff_with_reg.shape)
	print(OpInf_red_coeff_mild_reg.shape)

	
	rcParams['lines.linewidth'] = 0
	rc("figure", dpi=300)           # High-quality figure ("dots-per-inch")
	rc("text", usetex=True)         # Crisp axis ticks
	rc("font", family="serif")      # Crisp axis labels
	rc("legend", edgecolor='none')  # No boxes around legends
	rcParams["figure.figsize"] = (8, 5)
	rcParams.update({'font.size': 9})
	# rcParams['figure.conspreded_layout.use'] = True

	charcoal    = [0.1, 0.1, 0.1]
	color1      = '#D55E00'
	color2      = '#0072B2'
	
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
	pos2 = 21
	

	
	ax1.plot(t[:training_end], ref_coeff_training[:, pos1], linestyle='-', color=charcoal, lw=1.25, label='ref data (n = 3.8M)')
	ax1.plot(t, OpInf_red_coeff_with_reg[:, pos1], linestyle='-', color=color1, lw=1.25, label='std OpInf with reg (r = 24)')
	ax1.plot(t[:cutoff], OpInf_red_coeff_mild_reg[:cutoff, pos1], linestyle='-', color=color2, lw=1.25, label='OpInf with mild reg (r = 24)')
	ax1.set_xlabel('time [sec]')
	ax1.set_ylabel('red coeff for POD mode = 1')
	ax1.axvline(t[training_end - 1], lw=1.25, linestyle='--', color=charcoal)


	ax2.plot(t[:training_end], ref_coeff_training[:, pos2], linestyle='-', color=charcoal, lw=1.25, label='ref data (n = 3.8M)')
	ax2.plot(t, OpInf_red_coeff_with_reg[:, pos2], linestyle='-', color=color1, lw=1.25, label='std OpInf with reg (r = 24)')
	ax2.plot(t[:cutoff], OpInf_red_coeff_mild_reg[:cutoff, pos2], linestyle='-', color=color2, lw=1.25, label='OpInf with mild reg (r = 24)')
	ax2.set_xlabel('time [sec]')
	ax2.set_ylabel('red coeff for POD mode = 1')
	ax2.axvline(t[training_end - 1], lw=1.25, linestyle='--', color=charcoal)
	
	x_pos_all 	= np.array([0.0010, 0.0012, 0.0012474, 0.0014])
	labels 		= np.array([0.0010, 0.0012, 0.0012488, 0.0014])

	ax1.set_xticks(x_pos_all)
	ax1.set_xticklabels(labels)
	ax2.set_xticks(x_pos_all)
	ax2.set_xticklabels(labels)


	lines_labels = [ax1.get_legend_handles_labels()]
	lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

	legend = ax2.legend(lines, labels, loc = (0.02, -0.4), ncol=3, frameon=False)
	colors = [charcoal, color1, color2]
	for i, text in enumerate(legend.get_texts()):
		text.set_color(colors[i])

	tight_layout()

	fig_name = 'figures/RDE_red_coefficients.png'

	savefig(fig_name, pad_inches=3)
	close()