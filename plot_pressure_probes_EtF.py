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

	no_reg_cutoff = 200

	dt = (tf - ti)/441
	t  = np.arange(ti, tf, dt)
	t_OpInf = t


	FOM_data 				= np.load('OpInf_results/myprobes.npz')
	FOM_p 					= FOM_data['p_true']


	OpInf_file_with_reg 		= lambda x: 'OpInf_results/trace_steady_state_train_pred_with_reg_r' + str(x) + '.npz'
	SD_OpInf_data_with_reg 	= np.load(OpInf_file_with_reg(r1))
	SD_OpInf_p_with_reg 		= SD_OpInf_data_with_reg['p_rec']
	
	OpInf_file_no_reg 		= lambda x: 'OpInf_results/trace_steady_state_train_pred_mild_reg_r' + str(x) + '.npz'
	SD_OpInf_data_no_reg 	= np.load(OpInf_file_no_reg(r2))
	SD_OpInf_p_no_reg 		= SD_OpInf_data_no_reg['p_rec']

	
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
	pos2 = 3
	

	
	ax1.plot(t[:plotting_end], FOM_p[pos1, :plotting_end]/1e6, linestyle='-', color=charcoal, lw=1.25, label='full-order model (n = 3.8M)')
	ax1.plot(t[:training_end], SD_OpInf_p_with_reg[pos1, :training_end]/1e6, linestyle='-', color=color1, lw=1.25, label='std OpInf with reg (r = {})'.format(r1))
	ax1.plot(t[training_end:], SD_OpInf_p_with_reg[pos1, training_end:]/1e6, linestyle=linestyle3, color=color1, lw=1.25)
	ax1.plot(t[:no_reg_cutoff], SD_OpInf_p_no_reg[pos1, :no_reg_cutoff]/1e6, linestyle='-', color=color2, lw=1.25, label='OpInf with mild reg (r = {})'.format(r2))
	# ax1.plot(t[training_end:], SD_OpInf_p_no_reg[pos1, training_end:]/1e6, linestyle=linestyle4, color=color2, lw=1.25)
	ax1.set_xlabel('time [sec]')
	ax1.set_ylabel('Probe 1: pressure [MPa]')
	ax1.axvline(t[training_end - 1], lw=1.25, linestyle='--', color=charcoal)
	
	ax2.plot(t[:plotting_end], FOM_p[pos2, :plotting_end]/1e6, linestyle='-', color=charcoal, lw=1.25)
	ax2.plot(t[:training_end], SD_OpInf_p_with_reg[pos2, :training_end]/1e6, linestyle='-', color=color1, lw=1.25)
	ax2.plot(t[training_end:], SD_OpInf_p_with_reg[pos2, training_end:]/1e6, linestyle=linestyle3, color=color1, lw=1.25)
	ax2.plot(t[:no_reg_cutoff], SD_OpInf_p_no_reg[pos2, :no_reg_cutoff]/1e6, linestyle='-', color=color2, lw=1.25)
	# ax2.plot(t[training_end:], SD_OpInf_p_no_reg[pos2, training_end:]/1e6, linestyle=linestyle4, color=color2, lw=1.25)
	ax2.set_xlabel('time [sec]')
	ax2.set_ylabel('Probe 2: pressure [MPa]')
	ax2.axvline(t[training_end - 1], lw=1.25, linestyle='--', color=charcoal)

	# ax1.text(t[training_end - 20], 0.6, 'training end', color=charcoal)
	# ax2.text(t[training_end - 20], 0.6, 'training end', color=charcoal)


	# ax1.text(t[training_end - 1], 0.55, 'training end')


	# axins = zoomed_inset_axes(ax1, zoom=1.5, loc=4)
	# axins.plot(t[:plotting_end], FOM_p[pos1, :plotting_end]/1e6, linestyle='-', color=charcoal, lw=2.5)
	# axins.plot(t[:plotting_end], SD_OpInf_p_with_reg[pos1, :plotting_end]/1e6, linestyle=linestyle3, color=color1, lw=2.5)
	# axins.plot(t[:plotting_end], SD_OpInf_p_no_reg[pos1, :plotting_end]/1e6, linestyle=linestyle4, color=color2, lw=2.5)
	# axins.set_xlim(t[training_end - 1], t[-1])
	# axins.set_ylim(0.27, 0.47)
	# xticks(visible=False)
	# yticks(visible=False)
	# mark_inset(ax1, axins, loc1=2, loc2=3, fc="none", ec="0.5")
	

	# axins = zoomed_inset_axes(ax2, zoom=1.5, loc=4)
	# axins.plot(t[:plotting_end], FOM_p[pos2, :plotting_end]/1e6, linestyle='-', color=charcoal, lw=2.5)
	# axins.plot(t[:plotting_end], SD_OpInf_p_with_reg[pos2, :plotting_end]/1e6, linestyle=linestyle3, color=color1, lw=2.5)
	# axins.plot(t[:plotting_end], SD_OpInf_p_no_reg[pos2, :plotting_end]/1e6, linestyle=linestyle4, color=color2, lw=2.5)
	# axins.set_xlim(t[training_end - 1], t[-1])
	# axins.set_ylim(0.265, 0.462)
	# xticks(visible=False)
	# yticks(visible=False)
	# mark_inset(ax2, axins, loc1=2, loc2=3, fc="none", ec="0.5")

	

	
	# x_pos_all 	= np.array([0.0010, 0.0012, 0.0014, 0.0016])
	# labels 		= np.array([0.0010, 0.0012, 0.0014, ''])

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

	# show()
	fig_name = 'figures/RDE_pressure_probe_recon.png'

	savefig(fig_name, pad_inches=3)
	close()