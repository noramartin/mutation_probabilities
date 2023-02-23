#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import parameters_HP as paramHP
from functions_fullstructures.general_functions import correct_top_x_predicted, load_df_into_dict_of_dict
from HP_model.HPfunctions import *
from matplotlib.lines import Line2D
from functools import partial
import seaborn as sns
from scipy.stats import pearsonr

label_type_correlation, type_correlation, correlation_function = (r'fraction of top-30 $\phi_{qp}$' + '\npredicted correctly', 'top-30', partial(correct_top_x_predicted, x=30))
chosen_example_HP = 250
####
df_N_HP = pd.read_csv(paramHP.neutral_set_size_filename)
structure_vs_freq_estimateHP = {row['structure']: row['neutral set size']/float(paramHP.K**paramHP.L) for rowindex, row in df_N_HP.iterrows()}
structure_list = df_N_HP['structure'].tolist()[:]

df_phipq = pd.read_csv(paramHP.phipq_filename)
phi_vs_phi_pq_allstructuresHP = load_df_into_dict_of_dict(df_phipq, 'structure 1', 'structure 2', 'phi', structure_list)

###
f2, ax2 = plt.subplots(ncols=4, nrows = 3, figsize=(10, 7.5), gridspec_kw={'width_ratios':[1, 1, 1, 0.5]})
for rowindex, kbT in enumerate([0.1, 0.5, 1]):
   meanBoltzmann_neutral_set_filename = './GPmapdata_HP/HPmeanBoltzmann_neutral_set_L'+str(paramHP.L)+'_10kbT'+str(int(kbT * 10)) +'_'+str(paramHP.number_sequences_per_structure)+'.csv'
   ###############################################################################################
   ###############################################################################################
   print( 'load data - HP model', flush=True)
   ###############################################################################################
   ###############################################################################################
   try:
      df_Boltzmann_freq = pd.read_csv(meanBoltzmann_neutral_set_filename)  
      ph_vs_Boltzmann_freq_in_neutral_set_allstructuresHP = load_df_into_dict_of_dict(df_Boltzmann_freq, 'structure 1', 'structure 2', 'Boltzmann frequency', structure_list)
      df_corr_HP = pd.read_csv('./GPmapdata_HP/Psampling_phenotype_phipq_correlations'+type_correlation+'_'+str(paramHP.L)+'kbT' + str(kbT)+'.csv')
      ##############
      ph_HP = structure_list[chosen_example_HP]
      print(ph_HP)
      boltz_freq_list_HP = [ph_vs_Boltzmann_freq_in_neutral_set_allstructuresHP[ph_HP][s] for s in structure_list if s != ph_HP]
      freq_list_for_plot_HP = [structure_vs_freq_estimateHP[s] for s in structure_list if s != ph_HP]
      phipq_list_HP = [phi_vs_phi_pq_allstructuresHP[ph_HP][s] if s in phi_vs_phi_pq_allstructuresHP[ph_HP] else 0 for s in structure_list if s != ph_HP]
   except IOError:
      print('files not found for kbT', kbT)
      print('./GPmapdata_HP/Psampling_phenotype_phipq_correlations'+type_correlation+str(paramHP.L)+'kbT' + str(kbT)+'.csv')
      print(meanBoltzmann_neutral_set_filename, '\n\n')
      continue
   ###############################################################################################
   ###############################################################################################
   print( 'plot data', flush=True)
   ###############################################################################################
   ###############################################################################################

   axBoltz1, axfreq1, ax_stats1 = ax2[rowindex, 1], ax2[rowindex, 0], ax2[rowindex, 2]
   ax2[rowindex, 3].axis('off')
   custom_lines = [Line2D([0], [0], mfc='r', ls='', marker='o', label='average\nBoltzmann', mew=0, ms=5),
                   Line2D([0], [0], mfc='b', ls='', marker='o', label='phenotypic\nfrequency', mew=0, ms=5)]
   ax2[rowindex, 3].legend(handles=custom_lines)
   ###
   axBoltz1.scatter(boltz_freq_list_HP, phipq_list_HP, c='r', s=4, alpha=0.5)
   axBoltz1.set_ylabel(r'$\phi_{qp}$')#+ '\nfor' + r' $p=$ '+ str(ph_HP))
   axBoltz1.set_xlabel(r'$p_{qp}$')#+ '\nfor' + r' $p=$ '+ str(ph_HP))
   axBoltz1.set_xscale('log')
   axBoltz1.set_yscale('log')
   min_axlim = min([0.5,]+[min(g, phi_value) for (g, phi_value) in zip(boltz_freq_list_HP, phipq_list_HP) if g > 0 and phi_value > 0])
   axBoltz1.plot([min_axlim * 10.0**(-7), 2], [min_axlim * 10.0**(-7), 2], c='k', lw=0.5)
   axBoltz1.set_ylim(0.5 * min_axlim, 1.2)
   axBoltz1.set_xlim(0.5 * min_axlim, 1.2)
   ####
   print('HP freq', correct_top_x_predicted(freq_list_for_plot_HP, phipq_list_HP, 30)[0])
   axfreq1.scatter(freq_list_for_plot_HP, phipq_list_HP, c='b', s=4, alpha=0.5)
   axfreq1.set_ylabel(r'$\phi_{qp}$')# + '\n' + r'for $p=$ '+str(ph_HP))
   axfreq1.set_xlabel(r'$f_q$')
   axfreq1.set_xscale('log')
   axfreq1.set_yscale('log')
   min_axlim = min([0.5,]+[min(g, phi_value) for (g, phi_value) in zip(freq_list_for_plot_HP, phipq_list_HP) if g > 0 and phi_value > 0])
   axfreq1.plot([min_axlim * 10.0**(-7), 2], [min_axlim * 10.0**(-7), 2], c='k', lw=0.5)
   axfreq1.plot([1.0/(paramHP.K**paramHP.L), 1.0/(paramHP.K**paramHP.L)], [1.0/(paramHP.K**paramHP.L), 1.0/(paramHP.K**paramHP.L)])
   axfreq1.set_ylim(0.5 * min_axlim, 1.2)
   axfreq1.set_xlim(0.5 * min_axlim, 1.2)
   ###
   ax_stats1.scatter(np.arange(1, len(df_corr_HP['structure'].tolist()) + 1), 
                   df_corr_HP[type_correlation + ' correlation mean Boltzmann frequ. vs phiqp substitutions'].tolist(), 
                   c='r', label='Boltzmann', edgecolor=None, s=2)

   ax_stats1.scatter(np.arange(1, len(df_corr_HP['structure'].tolist()) + 1), 
                    df_corr_HP[type_correlation + ' correlation structure frequ. vs phiqp substitutions'].tolist(), 
                    c='b', label='frequency', edgecolor=None, s=2)
   ax_stats1.set_xlabel('rank of initial structure\n\n')
   ax_stats1.set_ylabel(label_type_correlation )

##########
for axindex, ax in np.ndenumerate(ax2[:, :3]):
   ax.annotate('ABCDEFGHIJ'[axindex[0] * 3 + axindex[1]], xy=(0.04, 0.86), xycoords='axes fraction', fontsize='large', fontweight='bold', size=13)  
 
f2.tight_layout()
f2.savefig('./plots/HP_vary_kbT_phi_pq_Boltzmann'+'examples'+str(chosen_example_HP)  +'.png', bbox_inches='tight')
del f2, ax2
plt.close('all')
###############################################################################################
###############################################################################################
print( 'plot neutral set size data', flush=True)
###############################################################################################
###############################################################################################
f, ax = plt.subplots(ncols = 3, nrows = 3, figsize=(10, 8.5))
for rowindex, kbT in enumerate([0.1, 0.5, 1]):
   for columnindex, mfe_included in enumerate([True, False]):
     if mfe_included:
        type_Boltzmann_average = '' #['', 'no_mfe', 'folding_seq_only']
     else:
        type_Boltzmann_average = 'no_mfe'
     meanBoltzmann_all_filenameHP = './GPmapdata_HP/HPmeanBoltzmann_gsamplet_L'+str(paramHP.L)+type_Boltzmann_average+'_100kbT'+str(int(100*kbT))  +'_'+str(paramHP.g_sample_size_meanBoltz_random_seq)+'.csv'
     df_Boltzmann_freq_HP = pd.read_csv(meanBoltzmann_all_filenameHP)  
     structure_vs_mean_Boltz_HP = {row['structure']: row['mean Boltzmann freq'] for rowindex, row in df_Boltzmann_freq_HP.iterrows()}
     #################
     ax[rowindex, columnindex].set_xscale('log')
     ax[rowindex, columnindex].set_yscale('log')
     ax[rowindex, columnindex].set_xlabel(r'phenotypic frequency $f_q$'+'\nof structure\n')
     if mfe_included:
         ax[rowindex, columnindex].set_ylabel(r'mean Boltzmann frequency $p_q$'+'\nin sequence sample\nfor '+r'$k_B T =$' + str(kbT))
     else:
         ax[rowindex, columnindex].set_ylabel(r'mean Boltzmann frequency $p_q$'+'\nas suboptimal structure\nin sequence sample\nfor '+r'$k_B T =$' + str(kbT))
     freq_list_plot, boltz_list_plot = [structure_vs_freq_estimateHP[s] for s in structure_list], [structure_vs_mean_Boltz_HP[s] for s in structure_list]
     ax[rowindex, columnindex].scatter(freq_list_plot, boltz_list_plot, s=3, c='grey', alpha=0.5)
     minxfreq = min([structure_vs_freq_estimateHP[s] for s in structure_list])
     maxxfreq = max([structure_vs_freq_estimateHP[s] for s in structure_list])
     minxBoltz, maxxBoltz = min([structure_vs_mean_Boltz_HP[s] for s in structure_list]), max([structure_vs_mean_Boltz_HP[s] for s in structure_list])
     ax[rowindex, columnindex].set_xlim(minxfreq * 0.5, maxxfreq *2)
     #ax[rowindex, columnindex].set_ylim(minxBoltz * 0.5, maxxBoltz *2)
     ax[rowindex, columnindex].set_title(r'Pearson corr. '+'{:.2f}'.format(pearsonr(freq_list_plot, boltz_list_plot)[0]))
     print(pearsonr(freq_list_plot, boltz_list_plot))
     #################
     gsampled_dist_filename = './GPmapdata_HP/HP_gsample_Boltzmann_dist_L'+str(paramHP.L)+'_100kbT'+str(int(100*kbT))  +'_'+str(paramHP.g_sample_size_hist)+'.npy'
     boltz_list = np.load(gsampled_dist_filename)
     ax[rowindex, -1].hist(boltz_list)
     ax[rowindex, -1].set_xlabel('Boltzmann frequency of\nmfe structure\nin arbitrary folding sequence')
     ax[rowindex, -1].set_ylabel('frequency')
f.tight_layout()
f.savefig('./plots/HP_varykbT_neutral_set_size_mean_Boltz'+type_Boltzmann_average+'.png', bbox_inches='tight')
plt.close('all')
del f, ax
