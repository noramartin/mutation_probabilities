#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import parameters_HP as paramHP
import parameters_fullstructures as paramRNA
from functions_fullstructures.general_functions import correct_top_x_predicted, load_df_into_dict_of_dict, logPearsonR_nonzero_values
from HP_model.HPfunctions import *
from copy import deepcopy
from matplotlib.lines import Line2D
from functools import partial
import RNA
from KC import calc_KC


#label_type_correlation, type_correlation, correlation_function = (r'fraction of top-30 $\phi_{qp}$' + '\npredicted correctly', 'top-30', partial(correct_top_x_predicted, x=30))
label_type_correlation, type_correlation, correlation_function = ('Pearson correlation\nbetween non-zero values', 'Pearson', logPearsonR_nonzero_values)

chosen_example_RNA, chosen_example_HP = 25, 250
####
###############################################################################################
###############################################################################################
print( 'load data - HP model', flush=True)
###############################################################################################
###############################################################################################
df_N_HP = pd.read_csv(paramHP.neutral_set_size_filename)
structure_vs_freq_estimateHP = {row['structure']: row['neutral set size']/float(paramHP.K**paramHP.L) for rowindex, row in df_N_HP.iterrows()}
structure_listHP = df_N_HP['structure'].tolist()[:]
df_phipq = pd.read_csv(paramHP.phipq_filename)
phi_vs_phi_pq_allstructuresHP = load_df_into_dict_of_dict(df_phipq, 'structure 1', 'structure 2', 'phi', structure_listHP)
df_Boltzmann_freq = pd.read_csv(paramHP.meanBoltzmann_neutral_set_filename)  
ph_vs_Boltzmann_freq_in_neutral_set_allstructuresHP = load_df_into_dict_of_dict(df_Boltzmann_freq, 'structure 1', 'structure 2', 'Boltzmann frequency', structure_listHP)
df_corr_HP = pd.read_csv('./GPmapdata_HP/Psampling_phenotype_phipq_correlations'+type_correlation+'_'+str(paramHP.L)+'kbT' + str(paramHP.kbT)+'.csv')
print('number of structures in HP data', df_N_HP.shape[0])
##############
for i in range(len(structure_listHP)-1): #test ordering
   assert structure_vs_freq_estimateHP[df_corr_HP['structure'].tolist()[i]] >= structure_vs_freq_estimateHP[df_corr_HP['structure'].tolist()[i+1]]
#############
ph_HP = structure_listHP[chosen_example_HP]
print(ph_HP)
boltz_freq_list_HP = [ph_vs_Boltzmann_freq_in_neutral_set_allstructuresHP[ph_HP][s] for s in structure_listHP if s != ph_HP]
freq_list_for_plot_HP = [structure_vs_freq_estimateHP[s] for s in structure_listHP if s != ph_HP]
phipq_list_HP = [phi_vs_phi_pq_allstructuresHP[ph_HP][s] if s in phi_vs_phi_pq_allstructuresHP[ph_HP] else 0 for s in structure_listHP if s != ph_HP]
#####
f, ax = plt.subplots(figsize=(2, 2))
CM = paramHP.contact_map_list[ph_HP - 1]
plot_structure(paramHP.contact_map_vs_updown[CM], ax)
f.tight_layout()
f.savefig('./plots/HPexample'+str(chosen_example_HP)+'.png', bbox_inches='tight')
plt.close('all')
"""
###############################################################################################
###############################################################################################
print( 'load data - full structures', flush=True)
###############################################################################################
###############################################################################################
######################
df_N_RNA = pd.read_csv(paramRNA.neutral_set_size_filename_sampling)
structure_vs_freq_estimateRNA = {row['structure']: row['frequency'] for rowindex, row in df_N_RNA.iterrows()}
df_phipq = pd.read_csv(paramRNA.phipq_filename_sampling)
df_NNE_dataRNA = pd.read_csv(paramRNA.Nresult_filename)
structure_vs_freq_estimateNNE = {row['structure']: row['Nmfe'] for rowindex, row in df_NNE_dataRNA.iterrows()}
structure_list_seq_sample = sorted(list(set(df_phipq['structure 1'].tolist()[:])), key=structure_vs_freq_estimateNNE.get, reverse=True)
phi_vs_phi_pq_allstructuresRNA = load_df_into_dict_of_dict(df_phipq, 'structure 1', 'structure 2', 'phi', structure_list_seq_sample)
df_Boltzmann_freq = pd.read_csv(paramRNA.meanBoltzmann_neutral_set_filename_sampling)  
ph_vs_Boltzmann_freq_in_neutral_set_allstructuresRNA = load_df_into_dict_of_dict(df_Boltzmann_freq, 'structure 1', 'structure 2', 'Boltzmann frequency', structure_list_seq_sample)
df_compRNA = pd.read_csv(paramRNA.compatible_filename_sampling)  
phi_vs_compatible_seq_metric_allstructures = load_df_into_dict_of_dict(df_compRNA, 'structure 1', 'structure 2', 'fraction comp seq', structure_list_seq_sample)
df_compRNA2 = pd.read_csv(paramRNA.compatible_filename_sampling_not_neutral_set)  
phi_vs_compatible_seq_metric_allstructures2 = load_df_into_dict_of_dict(df_compRNA2, 'structure 1', 'structure 2', 'fraction comp seq', structure_list_seq_sample)

df_corr_RNA = pd.read_csv('./GPmapdata_fullstructures/Psampling_phenotype_phipq_correlations'+type_correlation+str(paramRNA.sample_filename)+'.csv')
##############
for i in range(len(structure_list_seq_sample)-1): #test ordering
   assert structure_vs_freq_estimateNNE[df_corr_RNA['structure'].tolist()[i]] >= structure_vs_freq_estimateNNE[df_corr_RNA['structure'].tolist()[i+1]]
##############
ph_RNA = structure_list_seq_sample[chosen_example_RNA]
structure_list_all = list(set([s for s in ph_vs_Boltzmann_freq_in_neutral_set_allstructuresRNA[ph_RNA] if s not in ['|', '.']] + [s for s in phi_vs_phi_pq_allstructuresRNA[ph_RNA] if s not in ['|', '.']] + [s for s in structure_vs_freq_estimateRNA if s not in ['|', '.']]))
boltz_freq_list_RNA = [ph_vs_Boltzmann_freq_in_neutral_set_allstructuresRNA[ph_RNA][s] if s in ph_vs_Boltzmann_freq_in_neutral_set_allstructuresRNA[ph_RNA] else 0 for s in structure_list_all if s != ph_RNA]
freq_list_for_plot_RNA = [structure_vs_freq_estimateRNA[s] if s in structure_vs_freq_estimateRNA else 0 for s in structure_list_all if s != ph_RNA]
phipq_list_RNA = [phi_vs_phi_pq_allstructuresRNA[ph_RNA][s] if s in phi_vs_phi_pq_allstructuresRNA[ph_RNA] else 0 for s in structure_list_all if s != ph_RNA]
#####
if '.' * paramRNA.L_sampling in structure_list_all:
   print('unfolded structure is in structure list')
# the two-ground state structure appears as '.', but only in the frequency data
if '.' in structure_list_all:
   print('two-ground-state structure is in structure list')
assert '.' not in [y for x in ph_vs_Boltzmann_freq_in_neutral_set_allstructuresRNA for y in  ph_vs_Boltzmann_freq_in_neutral_set_allstructuresRNA[x]]
assert '.' not in [y for x in phi_vs_phi_pq_allstructuresRNA for y in  phi_vs_phi_pq_allstructuresRNA[x]]
######
RNA.PS_rna_plot(' '*len(ph_RNA), ph_RNA, './plots/RNAexample'+str(chosen_example_RNA)+'.ps')
############## overlap of sequence samples
# df_seq1 = pd.read_csv('./GPmapdata_fullstructures/sequence_sample'+'phi' +paramRNA.sample_filename+'mfe_unique.csv')
# df_seq2 = pd.read_csv('./GPmapdata_fullstructures/sequence_sample'+'Boltzmann' +paramRNA.sample_filename+'mfe_unique.csv')
# list_seq1 = [row['seq'] for rowi, row in df_seq1.iterrows() if row['structure'] == ph_RNA]
# list_seq2 = [row['seq'] for rowi, row in df_seq2.iterrows() if row['structure'] == ph_RNA]
# print('number of sequences in sample', len(list_seq1), len(list_seq2), 'combined length', len(set(list_seq1 + list_seq2)))

###############################################################################################
###############################################################################################
print( 'plot data', flush=True)
###############################################################################################
###############################################################################################
f2, ax2 = plt.subplots(ncols=4, nrows = 2, figsize=(10, 4), gridspec_kw={'width_ratios':[1, 1, 1, 0.5]})
axBoltz1, axfreq1, ax_stats1 = ax2[0, 1], ax2[0, 0], ax2[0, 2]
axBoltz2, axfreq2, ax_stats2 = ax2[1, 1], ax2[1, 0], ax2[1, 2]
ax2[0, 3].axis('off')
ax2[1, 3].axis('off')
custom_lines = [Line2D([0], [0], mfc='r', ls='', marker='o', label='average\nBoltzmann', mew=0, ms=5),
                Line2D([0], [0], mfc='b', ls='', marker='o', label='phenotypic\nfrequency', mew=0, ms=5)]
ax2[0, 3].legend(handles=custom_lines)
ax2[1, 3].legend(handles=custom_lines)      
###
#plot_loglog_plot_zero_lines_discrete_color(boltz_freq_list_RNA, phipq_list_RNA, axBoltz1, c='r')
axBoltz1.scatter(boltz_freq_list_RNA, phipq_list_RNA, c='r', s=4, alpha=0.5)
print('RNA Boltzmann', correct_top_x_predicted(boltz_freq_list_RNA, phipq_list_RNA, 30)[0])
axBoltz1.set_ylabel(r'$\phi_{qp}$')#+ '\nfor' + r' $p=$ '+ str(ph_RNA))
axBoltz1.set_xlabel(r'$p_{qp}$'+ '\n') #for' + r' $p=$ '+ str(ph_RNA))
axBoltz1.set_xscale('log')
axBoltz1.set_yscale('log')
min_axlim = min([0.5,]+[min(g, phi_value) for (g, phi_value) in zip(boltz_freq_list_RNA, phipq_list_RNA) if g > 0 and phi_value > 0])
axBoltz1.plot([min_axlim * 10.0**(-7), 2], [min_axlim * 10.0**(-7), 2], c='k', lw=0.5)
axBoltz1.set_ylim(0.5 * min_axlim, 1.2)
axBoltz1.set_xlim(0.5 * min_axlim, 1.2)
####
print('RNA freq', correct_top_x_predicted(freq_list_for_plot_RNA, phipq_list_RNA, 30)[0])
axfreq1.scatter(freq_list_for_plot_RNA, phipq_list_RNA, c='b', s=4, alpha=0.5)
axfreq1.set_ylabel(r'$\phi_{qp}$' )#+ '\n' + r'for $p=$ '+str(ph_RNA))
axfreq1.set_xlabel(r'$f_q$')
axfreq1.set_xscale('log')
axfreq1.set_yscale('log')
min_axlim = min([0.5,]+[min(g, phi_value) for (g, phi_value) in zip(freq_list_for_plot_RNA, phipq_list_RNA) if g > 0 and phi_value > 0])
axfreq1.plot([min_axlim * 10.0**(-7), 2], [min_axlim * 10.0**(-7), 2], c='k', lw=0.5)
axfreq1.plot([1.0/(paramRNA.K**paramRNA.L_sampling), 1.0/(paramRNA.K**paramRNA.L_sampling)], [1.0/(paramRNA.K**paramRNA.L_sampling), 1.0/(paramRNA.K**paramRNA.L_sampling)])
axfreq1.set_ylim(0.5 * min_axlim, 1.2)
axfreq1.set_xlim(0.5 * min_axlim, 1.2)
######
result, result, indices_true, indices_predicted = correct_top_x_predicted(freq_list_for_plot_RNA, phipq_list_RNA, x=30, return_indices=True)
high_phi_low_f = [structure for i, structure in enumerate([s for s in structure_list_all if s != ph_RNA]) if i in indices_true and i not in indices_predicted]
print('structures with deviation (freq)', high_phi_low_f)
shared_bps = [(s.count('(') + ph_RNA.count('(') - RNA.bp_distance(s, ph_RNA))/2 for s in high_phi_low_f]
print('shared bps (out of'+str(ph_RNA.count('('))+')', shared_bps)
######

###
ax_stats1.scatter(np.arange(1, len(df_corr_RNA['structure'].tolist()) + 1), 
                df_corr_RNA[type_correlation + ' correlation mean Boltzmann frequ. vs phiqp substitutions'].tolist(), c='r', label='Boltzmann', edgecolor=None, s=4)

ax_stats1.scatter(np.arange(1, len(df_corr_RNA['structure'].tolist()) + 1), 
                 df_corr_RNA[type_correlation + ' correlation structure frequ. vs phiqp substitutions'].tolist(), c='b', label='frequency', edgecolor=None, s=4)
ax_stats1.set_xlabel('rank of initial structure')
ax_stats1.set_ylabel(label_type_correlation )
##########
##########
print('HP Boltzmann', correct_top_x_predicted(boltz_freq_list_HP, phipq_list_HP, 30)[0])
structures_sorted_by_Boltz = sorted([s for s in structure_listHP if s != ph_HP], key=ph_vs_Boltzmann_freq_in_neutral_set_allstructuresHP[ph_HP].get, reverse=True)
for i in range(15):
   f, ax = plt.subplots(figsize=(2, 2))
   CM = paramHP.contact_map_list[structures_sorted_by_Boltz[i] - 1]
   plot_structure(paramHP.contact_map_vs_updown[CM], ax)
   f.tight_layout()
   f.savefig('./plots_examplesHP/HPexample'+str(i)+'_from'+str(ph_HP)+'.png', bbox_inches='tight')
   plt.close('all')
##########
axBoltz2.scatter(boltz_freq_list_HP, phipq_list_HP, c='r', s=4, alpha=0.5)
axBoltz2.set_ylabel(r'$\phi_{qp}$')#+ '\nfor' + r' $p=$ '+ str(ph_HP))
axBoltz2.set_xlabel(r'$p_{qp}$')#+ '\nfor' + r' $p=$ '+ str(ph_HP))
axBoltz2.set_xscale('log')
axBoltz2.set_yscale('log')
min_axlim = min([0.5,]+[min(g, phi_value) for (g, phi_value) in zip(boltz_freq_list_HP, phipq_list_HP) if g > 0 and phi_value > 0])
axBoltz2.plot([min_axlim * 10.0**(-7), 2], [min_axlim * 10.0**(-7), 2], c='k', lw=0.5)
axBoltz2.set_ylim(0.5 * min_axlim, 1.2)
axBoltz2.set_xlim(0.5 * min_axlim, 1.2)
####
print('HP freq', correct_top_x_predicted(freq_list_for_plot_HP, phipq_list_HP, 30)[0])
axfreq2.scatter(freq_list_for_plot_HP, phipq_list_HP, c='b', s=4, alpha=0.5)
axfreq2.set_ylabel(r'$\phi_{qp}$')# + '\n' + r'for $p=$ '+str(ph_HP))
axfreq2.set_xlabel(r'$f_q$')
axfreq2.set_xscale('log')
axfreq2.set_yscale('log')
min_axlim = min([0.5,]+[min(g, phi_value) for (g, phi_value) in zip(freq_list_for_plot_HP, phipq_list_HP) if g > 0 and phi_value > 0])
axfreq2.plot([min_axlim * 10.0**(-7), 2], [min_axlim * 10.0**(-7), 2], c='k', lw=0.5)
axfreq2.plot([1.0/(paramHP.K**paramHP.L), 1.0/(paramHP.K**paramHP.L)], [1.0/(paramHP.K**paramHP.L), 1.0/(paramHP.K**paramHP.L)])
axfreq2.set_ylim(0.5 * min_axlim, 1.2)
axfreq2.set_xlim(0.5 * min_axlim, 1.2)
###
print('number of Boltzmann structures', len(df_corr_HP['structure'].tolist()))
ax_stats2.scatter(np.arange(1, len(df_corr_HP['structure'].tolist()) + 1), 
                df_corr_HP[type_correlation + ' correlation mean Boltzmann frequ. vs phiqp substitutions'].tolist(), 
                c='r', label='Boltzmann', edgecolor=None, s=2)

ax_stats2.scatter(np.arange(1, len(df_corr_HP['structure'].tolist()) + 1), 
                 df_corr_HP[type_correlation + ' correlation structure frequ. vs phiqp substitutions'].tolist(), 
                 c='b', label='frequency', edgecolor=None, s=2)
ax_stats2.set_xlabel('rank of initial structure')
ax_stats2.set_ylabel(label_type_correlation )

##########
for axindex, ax in enumerate([axfreq1, axBoltz1, ax_stats1, axfreq2, axBoltz2, ax_stats2]):
  ax.annotate('ABCDEF'[axindex], xy=(0.04, 0.86), xycoords='axes fraction', fontsize='large', fontweight='bold', size=13)  
 
f2.tight_layout()
f2.savefig('./plots/HP_RNA_combined_phi_pq_Boltzmann'+'examples'+str(chosen_example_HP) + '_' + str(chosen_example_RNA) +type_correlation+'.png', bbox_inches='tight')
del f2, ax2
plt.close('all')

###############################################################################################
###############################################################################################
print( 'plot data - further examples for RNA', flush=True)
###############################################################################################
###############################################################################################
examples_to_plot = [structure_list_seq_sample[i] for i in range(0, len(structure_list_seq_sample), int(np.ceil(len(structure_list_seq_sample)/35)))]
nrows = int(np.ceil(len(examples_to_plot)/5))
print('plotting', len(examples_to_plot), 'out of', len(structure_list_seq_sample), 'phenotypes - in ', nrows, 'rows')
for plot_type in ['', 'plot_top30']:
  f, ax = plt.subplots(nrows=nrows, ncols=5, figsize=(11, 1.7 * nrows + 1.7))
  for plotindex, ph_RNA_alternative in enumerate(examples_to_plot):
     
     structure_list_all = list(set([s for s in ph_vs_Boltzmann_freq_in_neutral_set_allstructuresRNA[ph_RNA_alternative] if s not in ['|', '.']] + [s for s in phi_vs_phi_pq_allstructuresRNA[ph_RNA_alternative] if s not in ['|', '.']]))
     assert '|' not in structure_list_all
     boltz_freq_list_RNA = [ph_vs_Boltzmann_freq_in_neutral_set_allstructuresRNA[ph_RNA_alternative][s] if s in ph_vs_Boltzmann_freq_in_neutral_set_allstructuresRNA[ph_RNA_alternative] else 0 for s in structure_list_all if s != ph_RNA_alternative]
     phipq_list_RNA = [phi_vs_phi_pq_allstructuresRNA[ph_RNA_alternative][s] if s in phi_vs_phi_pq_allstructuresRNA[ph_RNA_alternative] else 0 for s in structure_list_all if s != ph_RNA_alternative]
     axindex = tuple((plotindex//5, plotindex%5))
     if plot_type == 'plot_top30':
        result, result, indices_true, indices_predicted = correct_top_x_predicted(boltz_freq_list_RNA, phipq_list_RNA, x=30, return_indices=True)
        top_boltz_listx, top_boltz_listy = zip(*[(boltz_freq_list_RNA[i], phipq_list_RNA[i]) for i in indices_predicted])
        top_phi_listx, top_phi_listy = zip(*[(boltz_freq_list_RNA[i], phipq_list_RNA[i]) for i in indices_true])
        ax[axindex].scatter(top_boltz_listx, top_boltz_listy, c='y', s=10, marker='x', zorder=2,linewidth=0.5)
        ax[axindex].scatter(top_phi_listx, top_phi_listy, c='g', s=10, marker='+', zorder=2,linewidth=0.5)
     ax[axindex].scatter(boltz_freq_list_RNA, phipq_list_RNA, c='r', s=3, alpha=0.4, zorder=1, edgecolors='none')
     ax[axindex].set_ylabel(r'$\phi_{qp}$')#+ '\nfor' + r' $p=$ '+ str(ph_RNA))
     ax[axindex].set_xlabel(r'$p_{qp}$'+ '\n') #for' + r' $p=$ '+ str(ph_RNA))
     ax[axindex].set_title(ph_RNA_alternative, fontsize='small')
     ax[axindex].set_xscale('log')
     ax[axindex].set_yscale('log')
     min_axlim = min([0.5,]+[min(g, phi_value) for (g, phi_value) in zip(boltz_freq_list_RNA, phipq_list_RNA) if g > 0 and phi_value > 0])
     ax[axindex].plot([min_axlim * 10.0**(-7), 2], [min_axlim * 10.0**(-7), 2], c='k', lw=0.5)
     ax[axindex].set_ylim(0.5 * min_axlim, 1.2)
     ax[axindex].set_xlim(0.5 * min_axlim, 1.2)
  if nrows * 5 > len(examples_to_plot):
     for plotindex in range(len(examples_to_plot), nrows * 5):
        ax[plotindex//5, plotindex%5].axis('off')
  f.tight_layout()
  f.savefig('./plots/RNA_Boltz_many_examples'+str(paramRNA.sample_filename)+plot_type+'.png', bbox_inches='tight', dpi=400)
  plt.close('all')
  del f, ax
  f, ax = plt.subplots(nrows=nrows, ncols=5, figsize=(11, 1.7 * nrows + 1.7))
  for plotindex, ph_RNA_alternative in enumerate(examples_to_plot):
     structure_list_all = list(set([s for s in structure_vs_freq_estimateRNA if s not in ['|', '.']] + [s for s in phi_vs_phi_pq_allstructuresRNA[ph_RNA_alternative] if s not in ['|', '.']]))
     assert '|' not in structure_list_all
     phipq_list_RNA = [phi_vs_phi_pq_allstructuresRNA[ph_RNA_alternative][s] if s in phi_vs_phi_pq_allstructuresRNA[ph_RNA_alternative] else 0 for s in structure_list_all if s != ph_RNA_alternative]
     freq_list_for_plot_RNA = [structure_vs_freq_estimateRNA[s] if s in structure_vs_freq_estimateRNA else 0 for s in structure_list_all if s != ph_RNA_alternative]
     axindex = tuple((plotindex//5, plotindex%5))
     if plot_type == 'plot_top30':
        result, result, indices_true, indices_predicted = correct_top_x_predicted(freq_list_for_plot_RNA, phipq_list_RNA, x=30, return_indices=True)
        top_boltz_listx, top_boltz_listy = zip(*[(freq_list_for_plot_RNA[i], phipq_list_RNA[i]) for i in indices_predicted])
        top_phi_listx, top_phi_listy = zip(*[(freq_list_for_plot_RNA[i], phipq_list_RNA[i]) for i in indices_true])
        ax[axindex].scatter(top_boltz_listx, top_boltz_listy, c='y', s=10, marker='x', zorder=2,linewidth=0.5)
        ax[axindex].scatter(top_phi_listx, top_phi_listy, c='g', s=10, marker='+', zorder=2,linewidth=0.5)
     ax[axindex].scatter(freq_list_for_plot_RNA, phipq_list_RNA, c='b', s=3, alpha=0.3, zorder=1, edgecolors='none')
     ax[axindex].set_ylabel(r'$\phi_{qp}$')#+ '\nfor' + r' $p=$ '+ str(ph_RNA))
     ax[axindex].set_xlabel(r'$f_{q}$'+ '\n') #for' + r' $p=$ '+ str(ph_RNA))
     ax[axindex].set_title(ph_RNA_alternative, fontsize='small')
     ax[axindex].set_xscale('log')
     ax[axindex].set_yscale('log')
     min_axlim = min([0.5,]+[min(g, phi_value) for (g, phi_value) in zip(freq_list_for_plot_RNA, phipq_list_RNA) if g > 0 and phi_value > 0])
     ax[axindex].plot([min_axlim * 10.0**(-7), 2], [min_axlim * 10.0**(-7), 2], c='k', lw=0.5)
     ax[axindex].set_ylim(0.5 * min_axlim, 1.2)
     ax[axindex].set_xlim(0.5 * min_axlim, 1.2)
  if nrows * 5 > len(examples_to_plot):
     for plotindex in range(len(examples_to_plot), nrows * 5):
        ax[plotindex//5, plotindex%5].axis('off')
  f.tight_layout()
  f.savefig('./plots/RNA_freq_many_examples'+str(paramRNA.sample_filename)+plot_type+'.png', bbox_inches='tight', dpi=400)
  plt.close('all')
  del f, ax

###############################################################################################
###############################################################################################
print( 'plot data - similarity measures for RNA', flush=True)
###############################################################################################
###############################################################################################
## comparison
df_corr_RNA_2 = pd.read_csv('./GPmapdata_fullstructures/Psampling_phenotype_phipq_correlations'+type_correlation+str(paramRNA.sample_filename)+'_with_sim.csv')
f, ax = plt.subplots(ncols=2, figsize=(7, 3), gridspec_kw={'width_ratios':[1,0.5]})
type_similarity_vs_colour = {'mean Boltzmann frequ.': 'b', 'structure frequ.': 'r', 
                            'fraction sequence compatibility': 'c', '(1 - conditional complexity)': 'lime', '(1 - bp.dist)': 'orange', 'fraction sequence compatibility  (not neutral set)': 'g'}
type_similarity_vs_label = {'mean Boltzmann frequ.': 'Boltzmann', 'structure frequ.': 'frequency', 
                            'fraction sequence compatibility': 'compatible seq. (neutral set)', '(1 - conditional complexity)': 'AIT', '(1 - bp.dist)': 'bp dist.', 'fraction sequence compatibility  (not neutral set)': 'compatible seq. (all)'}

ax[1].axis('off')
type_similarity_list = ['mean Boltzmann frequ.', 'structure frequ.', '(1 - bp.dist)', 
                        'fraction sequence compatibility', 'fraction sequence compatibility  (not neutral set)', '(1 - conditional complexity)']
custom_lines = [Line2D([0], [0], mfc=type_similarity_vs_colour[label], ls='', marker='o', label=type_similarity_vs_label[label], mew=0, ms=5) for label in type_similarity_list]
ax[1].legend(handles=custom_lines)
for i, type_similarity in enumerate(type_similarity_list):
   if 'not neutral set' not in type_similarity:
     columnlabel = type_correlation + ' correlation '+type_similarity+' vs phiqp substitutions'
   else:
     columnlabel = type_correlation + ' correlation '+type_similarity.split('(')[0].strip()+' vs phiqp substitutions' + ' (' + type_similarity.split('(')[1]
   ax[0].scatter(np.arange(1, len(df_corr_RNA_2['structure'].tolist()) + 1) + (i - 2.5)/10.0, 
                df_corr_RNA_2[columnlabel].tolist(), 
                label=type_similarity_vs_label[type_similarity], edgecolor=None, s=6, c= type_similarity_vs_colour[type_similarity], alpha=0.88)#, ls='-')
#ax.legend()
ax[0].set_xlabel('rank of initial structure')
ax[0].set_ylabel(label_type_correlation )
f.tight_layout()
f.savefig('./plots/RNA_similarity_measures'+str(paramRNA.sample_filename)+type_correlation+'.png', bbox_inches='tight', dpi=200)
plt.close('all')
del f, ax
###############################
for plot_type in ['', 'plot_top30']:
  #### fraction compatible sequences
  f, ax = plt.subplots(nrows=nrows, ncols=5, figsize=(11, 1.7 * nrows + 1.7))
  for plotindex, ph_RNA_alternative in enumerate(examples_to_plot):
     structure_list_all = [s for s in phi_vs_phi_pq_allstructuresRNA[ph_RNA_alternative] if s not in ['|', '.']]
     assert '|' not in structure_list_all
     phipq_list_RNA = [phi_vs_phi_pq_allstructuresRNA[ph_RNA_alternative][s] if s in phi_vs_phi_pq_allstructuresRNA[ph_RNA_alternative] else 0 for s in structure_list_all if s != ph_RNA_alternative]
     comp_seq_list_for_plot_RNA = [phi_vs_compatible_seq_metric_allstructures[ph_RNA_alternative][s] if s in phi_vs_compatible_seq_metric_allstructures[ph_RNA_alternative] else 0 for s in structure_list_all if s != ph_RNA_alternative]
     axindex = tuple((plotindex//5, plotindex%5))
     if plot_type == 'plot_top30':
        result, result, indices_true, indices_predicted = correct_top_x_predicted(comp_seq_list_for_plot_RNA, phipq_list_RNA, x=30, return_indices=True)
        top_boltz_listx, top_boltz_listy = zip(*[(comp_seq_list_for_plot_RNA[i], phipq_list_RNA[i]) for i in indices_predicted])
        top_phi_listx, top_phi_listy = zip(*[(comp_seq_list_for_plot_RNA[i], phipq_list_RNA[i]) for i in indices_true])
        ax[axindex].scatter(top_boltz_listx, top_boltz_listy, c='y', s=10, marker='x', zorder=2,linewidth=0.5)
        ax[axindex].scatter(top_phi_listx, top_phi_listy, c='g', s=10, marker='+', zorder=2,linewidth=0.5)
     ax[axindex].scatter(comp_seq_list_for_plot_RNA, phipq_list_RNA, c='grey', s=3, alpha=0.3)
     ax[axindex].set_ylabel(r'$\phi_{qp}$')#+ '\nfor' + r' $p=$ '+ str(ph_RNA))
     ax[axindex].set_xlabel('fraction of\nsequences compatible') #for' + r' $p=$ '+ str(ph_RNA))
     ax[axindex].set_title(ph_RNA_alternative, fontsize='small')
     ax[axindex].set_xscale('log')
     ax[axindex].set_yscale('log')
     ######
     structure_list_all_without_ph = [s for s in structure_list_all if s != ph_RNA_alternative]
     examples_with_big_diff = sorted([(comp_seq_list_for_plot_RNA[i]/phipq_list_RNA[i] > 10**3, s) for i, s in enumerate(structure_list_all_without_ph) if s != ph_RNA_alternative and phipq_list_RNA[i] > 0], reverse=True)
     print(ph_RNA_alternative, 'structures with deviation (compatible seq)', [s[1] for s in examples_with_big_diff[:min(10, len(examples_with_big_diff))]])
     examples_without_big_diff = [s for i, s in enumerate(structure_list_all_without_ph) if s != ph_RNA_alternative and comp_seq_list_for_plot_RNA[i] > 0 and 0.1 < phipq_list_RNA[i]/comp_seq_list_for_plot_RNA[i]  < 10]
     print(ph_RNA_alternative, 'structures without deviation (compatible seq)', examples_without_big_diff[:min(10, len(examples_without_big_diff))], '\n---\n')
     ######
     assert max(comp_seq_list_for_plot_RNA) <= 1 and max(phipq_list_RNA) < 1
     min_axlim = min([0.5,]+[min(g, phi_value) for (g, phi_value) in zip(comp_seq_list_for_plot_RNA, phipq_list_RNA) if g > 0 and phi_value > 0])
     ax[axindex].plot([min_axlim * 10.0**(-7), 2], [min_axlim * 10.0**(-7), 2], c='k', lw=0.5)
     ax[axindex].set_ylim(0.5 * min_axlim, 4)
     ax[axindex].set_xlim(0.5 * min_axlim, 4)
  if nrows * 5 > len(examples_to_plot):
     for plotindex in range(len(examples_to_plot), nrows * 5):
        ax[plotindex//5, plotindex%5].axis('off')
  f.tight_layout()
  f.savefig('./plots/RNA_comp_seq_many_examples'+str(paramRNA.sample_filename)+plot_type+'.png', bbox_inches='tight', dpi=200)
  plt.close('all')
  del f, ax
  #### fraction compatible sequences beyond neutral set
  f, ax = plt.subplots(nrows=nrows, ncols=5, figsize=(11, 1.7 * nrows + 1.7))
  for plotindex, ph_RNA_alternative in enumerate(examples_to_plot):
     structure_list_all = [s for s in phi_vs_phi_pq_allstructuresRNA[ph_RNA_alternative] if s not in ['|', '.']]
     assert '|' not in structure_list_all
     phipq_list_RNA = [phi_vs_phi_pq_allstructuresRNA[ph_RNA_alternative][s] if s in phi_vs_phi_pq_allstructuresRNA[ph_RNA_alternative] else 0 for s in structure_list_all if s != ph_RNA_alternative]
     comp_seq_list_for_plot_RNA = [phi_vs_compatible_seq_metric_allstructures2[ph_RNA_alternative][s] if s in phi_vs_compatible_seq_metric_allstructures2[ph_RNA_alternative] else 0 for s in structure_list_all if s != ph_RNA_alternative]
     axindex = tuple((plotindex//5, plotindex%5))
     if plot_type == 'plot_top30':
        result, result, indices_true, indices_predicted = correct_top_x_predicted(comp_seq_list_for_plot_RNA, phipq_list_RNA, x=30, return_indices=True)
        top_boltz_listx, top_boltz_listy = zip(*[(comp_seq_list_for_plot_RNA[i], phipq_list_RNA[i]) for i in indices_predicted])
        top_phi_listx, top_phi_listy = zip(*[(comp_seq_list_for_plot_RNA[i], phipq_list_RNA[i]) for i in indices_true])
        ax[axindex].scatter(top_boltz_listx, top_boltz_listy, c='y', s=10, marker='x', zorder=2,linewidth=0.5)
        ax[axindex].scatter(top_phi_listx, top_phi_listy, c='g', s=10, marker='+', zorder=2,linewidth=0.5)
     ax[axindex].scatter(comp_seq_list_for_plot_RNA, phipq_list_RNA, c='grey', s=3, alpha=0.3)
     ax[axindex].set_ylabel(r'$\phi_{qp}$')#+ '\nfor' + r' $p=$ '+ str(ph_RNA))
     ax[axindex].set_xlabel('fraction of\nsequences compatible') #for' + r' $p=$ '+ str(ph_RNA))
     ax[axindex].set_title(ph_RNA_alternative, fontsize='small')
     ax[axindex].set_xscale('log')
     ax[axindex].set_yscale('log')
     ######
     structure_list_all_without_ph = [s for s in structure_list_all if s != ph_RNA_alternative]
     examples_with_big_diff = sorted([(comp_seq_list_for_plot_RNA[i]/phipq_list_RNA[i] > 10**3, s) for i, s in enumerate(structure_list_all_without_ph) if s != ph_RNA_alternative and phipq_list_RNA[i] > 0], reverse=True)
     print(ph_RNA_alternative, 'structures with deviation (compatible seq)', [s[1] for s in examples_with_big_diff[:min(10, len(examples_with_big_diff))]])
     examples_without_big_diff = [s for i, s in enumerate(structure_list_all_without_ph) if s != ph_RNA_alternative and comp_seq_list_for_plot_RNA[i] > 0 and  0.1 < phipq_list_RNA[i]/comp_seq_list_for_plot_RNA[i]  < 10]
     print(ph_RNA_alternative, 'structures without deviation (compatible seq 2)', examples_without_big_diff[:min(10, len(examples_without_big_diff))], '\n---\n')
     ######
     assert max(comp_seq_list_for_plot_RNA) <= 1 and max(phipq_list_RNA) < 1
     min_axlim = min([0.5,]+[min(g, phi_value) for (g, phi_value) in zip(comp_seq_list_for_plot_RNA, phipq_list_RNA) if g > 0 and phi_value > 0])
     ax[axindex].plot([min_axlim * 10.0**(-7), 2], [min_axlim * 10.0**(-7), 2], c='k', lw=0.5)
     ax[axindex].set_ylim(0.5 * min_axlim, 4)
     ax[axindex].set_xlim(0.5 * min_axlim, 4)
  if nrows * 5 > len(examples_to_plot):
     for plotindex in range(len(examples_to_plot), nrows * 5):
        ax[plotindex//5, plotindex%5].axis('off')
  f.tight_layout()
  f.savefig('./plots/RNA_comp_seq_not_neutral_set_many_examples'+str(paramRNA.sample_filename)+plot_type+'.png', bbox_inches='tight', dpi=200)
  plt.close('all')
  del f, ax
  #### bp dist
  f, ax = plt.subplots(nrows=nrows, ncols=5, figsize=(11, 1.7 * nrows + 1.7))
  for plotindex, ph_RNA_alternative in enumerate(examples_to_plot):
     structure_list_all = [s for s in phi_vs_phi_pq_allstructuresRNA[ph_RNA_alternative] if s not in ['|', '.']]
     assert '|' not in structure_list_all
     phipq_list_RNA = [phi_vs_phi_pq_allstructuresRNA[ph_RNA_alternative][s] if s in phi_vs_phi_pq_allstructuresRNA[ph_RNA_alternative] else 0 for s in structure_list_all if s != ph_RNA_alternative]
     bp_dist_list_for_plot_RNA = [RNA.bp_distance(s, ph_RNA_alternative) for s in structure_list_all if s != ph_RNA_alternative]
     axindex = tuple((plotindex//5, plotindex%5))
     if plot_type == 'plot_top30':
        result, result, indices_true, indices_predicted = correct_top_x_predicted([2 * len(ph_RNA_alternative) - b for b in bp_dist_list_for_plot_RNA], phipq_list_RNA, x=30, return_indices=True)
        top_boltz_listx, top_boltz_listy = zip(*[(bp_dist_list_for_plot_RNA[i], phipq_list_RNA[i]) for i in indices_predicted])
        top_phi_listx, top_phi_listy = zip(*[(bp_dist_list_for_plot_RNA[i], phipq_list_RNA[i]) for i in indices_true])
        ax[axindex].scatter(top_boltz_listx, top_boltz_listy, c='y', s=10, marker='x', zorder=2,linewidth=0.5)
        ax[axindex].scatter(top_phi_listx, top_phi_listy, c='g', s=10, marker='+', zorder=2,linewidth=0.5)
     ax[axindex].scatter(bp_dist_list_for_plot_RNA, phipq_list_RNA, c='grey', s=3, alpha=0.3)
     ax[axindex].set_ylabel(r'$\phi_{qp}$')#+ '\nfor' + r' $p=$ '+ str(ph_RNA))
     ax[axindex].set_xlabel('base pair distance\n' + r'between $p$ and $q$') #for' + r' $p=$ '+ str(ph_RNA))
     ax[axindex].set_title(ph_RNA_alternative, fontsize='small')
     ax[axindex].set_yscale('log')
     ######
     structure_list_all_without_ph = [s for s in structure_list_all if s != ph_RNA_alternative]
     examples_with_big_diff = [s for i, s in enumerate(structure_list_all_without_ph) if s != ph_RNA_alternative and phipq_list_RNA[i] < 10.0**(-4) and bp_dist_list_for_plot_RNA[i] == 10]
     print(ph_RNA_alternative, 'structures with deviation (bp dist)', examples_with_big_diff[:min(10, len(examples_with_big_diff))])
     ######
     min_axlim = min([0.5,]+[phi_value for phi_value in phipq_list_RNA if  phi_value > 0])
     ax[axindex].set_ylim(0.5 * min_axlim, 4)
  if nrows * 5 > len(examples_to_plot):
     for plotindex in range(len(examples_to_plot), nrows * 5):
        ax[plotindex//5, plotindex%5].axis('off')
  f.tight_layout()
  f.savefig('./plots/RNA_dist_seq_many_examples'+str(paramRNA.sample_filename)+plot_type+'.png', bbox_inches='tight', dpi=200)
  plt.close('all')
  del f, ax
  #### Dingle pred
  
  def dotbracket_to_bin(db_string):
    return ''.join([{'(': '10', ')': '01', '.': '00'}[c] for c in db_string])

  f, ax = plt.subplots(nrows=nrows, ncols=5, figsize=(11, 1.7 * nrows + 1.7))
  for plotindex, ph_RNA_alternative in enumerate(examples_to_plot):
     structure_list_all = [s for s in phi_vs_phi_pq_allstructuresRNA[ph_RNA_alternative] if s not in ['|', '.']]
     assert '|' not in structure_list_all
     phipq_list_RNA = [phi_vs_phi_pq_allstructuresRNA[ph_RNA_alternative][s] if s in phi_vs_phi_pq_allstructuresRNA[ph_RNA_alternative] else 0 for s in structure_list_all if s != ph_RNA_alternative]
     Dingle_pred_list = [calc_KC(dotbracket_to_bin(s+ph_RNA_alternative)) - calc_KC(dotbracket_to_bin(ph_RNA_alternative)) for s in structure_list_all if s != ph_RNA_alternative]
     axindex = tuple((plotindex//5, plotindex%5))
     if plot_type == 'plot_top30':
        result, result, indices_true, indices_predicted = correct_top_x_predicted([1 + max(Dingle_pred_list) - b for b in Dingle_pred_list], phipq_list_RNA, x=30, return_indices=True)
        top_boltz_listx, top_boltz_listy = zip(*[(Dingle_pred_list[i], phipq_list_RNA[i]) for i in indices_predicted])
        top_phi_listx, top_phi_listy = zip(*[(Dingle_pred_list[i], phipq_list_RNA[i]) for i in indices_true])
        ax[axindex].scatter(top_boltz_listx, top_boltz_listy, c='y', s=10, marker='x', zorder=2,linewidth=0.5)
        ax[axindex].scatter(top_phi_listx, top_phi_listy, c='g', s=10, marker='+', zorder=2,linewidth=0.5)
     ax[axindex].scatter(Dingle_pred_list, phipq_list_RNA, c='grey', s=3, alpha=0.3)
     ax[axindex].set_ylabel(r'$\phi_{qp}$')#+ '\nfor' + r' $p=$ '+ str(ph_RNA))
     ax[axindex].set_xlabel('AIT approach') #for' + r' $p=$ '+ str(ph_RNA))
     ax[axindex].set_title(ph_RNA_alternative, fontsize='small')
     ax[axindex].set_yscale('log')
     min_axlim = min([0.5,]+[phi_value for phi_value in phipq_list_RNA if  phi_value > 0])
     ax[axindex].set_ylim(0.5 * min_axlim, 4)
  if nrows * 5 > len(examples_to_plot):
     for plotindex in range(len(examples_to_plot), nrows * 5):
        ax[plotindex//5, plotindex%5].axis('off')
  f.tight_layout()
  f.savefig('./plots/RNA_Dingle_pred_many_examples'+str(paramRNA.sample_filename)+plot_type+'.png', bbox_inches='tight', dpi=200)
  plt.close('all')
  del f, ax
"""
###############################################################################################
###############################################################################################
print( 'plot data - further examples for HP', flush=True)
###############################################################################################
###############################################################################################
int_to_direction = {0: 'r', 1: 'l', 2: 'u', 3: 'd'}
examples_to_plot = [structure_listHP[i] for i in range(0, len(structure_listHP), int(np.ceil(len(structure_listHP)/35)))]
nrows = int(np.ceil(len(examples_to_plot)/5))
print('plotting', len(examples_to_plot), 'out of', len(structure_listHP), 'phenotypes - in ', nrows, 'rows')
for plot_type in ['', 'plot_top30']:
  f, ax = plt.subplots(nrows=nrows, ncols=5, figsize=(11, 1.7 * nrows + 1.7))
  for plotindex, ph_HP_alternative in enumerate(examples_to_plot):
     ph_description = ''.join([int_to_direction[int(i)] for i in paramHP.contact_map_vs_updown[paramHP.contact_map_list[ph_HP_alternative - 1]]])
     boltz_freq_list_HP  = [ph_vs_Boltzmann_freq_in_neutral_set_allstructuresHP[ph_HP_alternative][s] if s in ph_vs_Boltzmann_freq_in_neutral_set_allstructuresHP[ph_HP_alternative] else 0 for s in structure_listHP if s != ph_HP_alternative]
     phipq_list_HP  = [phi_vs_phi_pq_allstructuresHP[ph_HP_alternative][s] if s in phi_vs_phi_pq_allstructuresHP[ph_HP_alternative] else 0 for s in structure_listHP if s != ph_HP_alternative]
     axindex = tuple((plotindex//5, plotindex%5))
     if plot_type == 'plot_top30':
        result, result, indices_true, indices_predicted = correct_top_x_predicted(boltz_freq_list_HP, phipq_list_HP, x=30, return_indices=True)
        if len(indices_true) > 0:
           top_boltz_listx, top_boltz_listy = zip(*[(boltz_freq_list_HP[i], phipq_list_HP[i]) for i in indices_predicted])
           top_phi_listx, top_phi_listy = zip(*[(boltz_freq_list_HP[i], phipq_list_HP[i]) for i in indices_true])
           ax[axindex].scatter(top_boltz_listx, top_boltz_listy, c='y', s=10, marker='x', zorder=2,linewidth=0.5)
           ax[axindex].scatter(top_phi_listx, top_phi_listy, c='g', s=10, marker='+', zorder=2,linewidth=0.5)
     if len([s for s in phipq_list_HP if s > 0]) == 0:
        print('non non-zero phipq values for', ph_description)
     ax[axindex].scatter(boltz_freq_list_HP , phipq_list_HP , c='r', s=3, alpha=0.4)
     ax[axindex].set_ylabel(r'$\phi_{qp}$')#+ '\nfor' + r' $p=$ '+ str(ph_RNA))
     ax[axindex].set_xlabel(r'$p_{qp}$'+ '\n') #for' + r' $p=$ '+ str(ph_RNA))
     ax[axindex].set_title(ph_description, fontsize='small')
     ax[axindex].set_xscale('log')
     ax[axindex].set_yscale('log')
     min_axlim = min([0.5,]+[min(g, phi_value) for (g, phi_value) in zip(boltz_freq_list_HP , phipq_list_HP ) if g > 0 and phi_value > 0])
     ax[axindex].plot([min_axlim * 10.0**(-7), 2], [min_axlim * 10.0**(-7), 2], c='k', lw=0.5)
     ax[axindex].set_ylim(0.5 * min_axlim, 1.2)
     ax[axindex].set_xlim(0.5 * min_axlim, 1.2)
  if nrows * 5 > len(examples_to_plot):
     for plotindex in range(len(examples_to_plot), nrows * 5):
        ax[plotindex//5, plotindex%5].axis('off')
  f.tight_layout()
  f.savefig('./plots/HP_Boltz_many_examples'+str(paramHP.L)+'kbT' + str(int(10 *paramHP.kbT)) +plot_type+'.png', bbox_inches='tight', dpi=200)
  plt.close('all')
  del f, ax
  f, ax = plt.subplots(nrows=nrows, ncols=5, figsize=(11, 1.7 * nrows + 1.7))
  for plotindex, ph_HP_alternative in enumerate(examples_to_plot):
     ph_description = ''.join([int_to_direction[int(i)] for i in paramHP.contact_map_vs_updown[paramHP.contact_map_list[ph_HP_alternative -1]]])
     phipq_list_HP = [phi_vs_phi_pq_allstructuresHP[ph_HP_alternative][s] if s in phi_vs_phi_pq_allstructuresHP[ph_HP_alternative] else 0 for s in structure_listHP if s != ph_HP_alternative]
     freq_list_for_plot_HP  = [structure_vs_freq_estimateHP[s] if s in structure_vs_freq_estimateHP else 0 for s in structure_listHP if s != ph_HP_alternative]
     axindex = tuple((plotindex//5, plotindex%5))
     if plot_type == 'plot_top30':
        result, result, indices_true, indices_predicted = correct_top_x_predicted(freq_list_for_plot_HP, phipq_list_HP, x=30, return_indices=True)
        if len(indices_true) > 0:
           top_boltz_listx, top_boltz_listy = zip(*[(freq_list_for_plot_HP[i], phipq_list_HP[i]) for i in indices_predicted])
           top_phi_listx, top_phi_listy = zip(*[(freq_list_for_plot_HP[i], phipq_list_HP[i]) for i in indices_true])
           ax[axindex].scatter(top_boltz_listx, top_boltz_listy, c='y', s=10, marker='x', zorder=2,linewidth=0.5)
           ax[axindex].scatter(top_phi_listx, top_phi_listy, c='g', s=10, marker='+', zorder=2,linewidth=0.5)
     ax[axindex].scatter(freq_list_for_plot_HP , phipq_list_HP , c='b', s=3, alpha=0.3)
     ax[axindex].set_ylabel(r'$\phi_{qp}$')#+ '\nfor' + r' $p=$ '+ str(ph_RNA))
     ax[axindex].set_xlabel(r'$f_{q}$'+ '\n') #for' + r' $p=$ '+ str(ph_RNA))
     ax[axindex].set_title(ph_description, fontsize='small')
     ax[axindex].set_xscale('log')
     ax[axindex].set_yscale('log')
     min_axlim = min([0.5,]+[min(g, phi_value) for (g, phi_value) in zip(freq_list_for_plot_HP , phipq_list_HP ) if g > 0 and phi_value > 0])
     ax[axindex].plot([min_axlim * 10.0**(-7), 2], [min_axlim * 10.0**(-7), 2], c='k', lw=0.5)
     ax[axindex].set_ylim(0.5 * min_axlim, 1.2)
     ax[axindex].set_xlim(0.5 * min_axlim, 1.2)
  if nrows * 5 > len(examples_to_plot):
     for plotindex in range(len(examples_to_plot), nrows * 5):
        ax[plotindex//5, plotindex%5].axis('off')
  f.tight_layout()
  f.savefig('./plots/HP_freq_many_examples'+str(paramHP.L)+'kbT' + str(int(10 *paramHP.kbT))+plot_type+'.png', bbox_inches='tight', dpi=200)
  plt.close('all')
  del f, ax
###############################################################################################
###############################################################################################
print( 'plot data - similarity measures for HP', flush=True)
###############################################################################################
###############################################################################################
## comparison
f, ax = plt.subplots(ncols=2, figsize=(7, 3), gridspec_kw={'width_ratios':[1,0.5]})
type_similarity_vs_colour = {'mean Boltzmann frequ.': 'b', 'structure frequ.': 'r', 
                            '(1 - conditional complexity)': 'teal', 'CMO': 'lime', 'overlap of core positions': 'orange'}
type_similarity_vs_label = {'mean Boltzmann frequ.': 'Boltzmann', 'structure frequ.': 'frequency', 
                            '(1 - conditional complexity)': 'AIT', 'CMO': 'CMO', 'overlap of core positions': 'surface/core pattern'}
type_similarity_list = ['mean Boltzmann frequ.', 'structure frequ.', 
                         'CMO', 'overlap of core positions', '(1 - conditional complexity)']
ax[1].axis('off')
custom_lines = [Line2D([0], [0], mfc=type_similarity_vs_colour[label], ls='', marker='o', label=type_similarity_vs_label[label], mew=0, ms=5, alpha=0.7) for label in type_similarity_list]
ax[1].legend(handles=custom_lines, loc='center')
for i, type_similarity in enumerate(type_similarity_list):
   ax[0].scatter(np.arange(1, len(df_corr_HP['structure'].tolist()) + 1) + (i - 2)/8.0, 
                df_corr_HP[type_correlation + ' correlation '+type_similarity+' vs phiqp substitutions'].tolist(), 
                label=type_similarity_vs_label[type_similarity], edgecolor=None, s=0.5, edgecolors='none', alpha=0.7, c= type_similarity_vs_colour[type_similarity])#, ls='-')
#ax.legend()
ax[0].set_xlabel('rank of initial structure')
ax[0].set_ylabel(label_type_correlation )
f.tight_layout()
f.savefig('./plots/HP_similarity_measures'+str(paramRNA.sample_filename)+type_correlation+'.png', bbox_inches='tight', dpi=500)
plt.close('all')
del f, ax
### CMO
f, ax = plt.subplots(nrows=nrows, ncols=5, figsize=(11, 1.7 * nrows + 1.7))
for plotindex, ph_HP_alternative in enumerate(examples_to_plot):
   ph_description = ''.join([int_to_direction[int(i)] for i in paramHP.contact_map_vs_updown[paramHP.contact_map_list[ph_HP_alternative -1]]])
   phipq_list_HP = [phi_vs_phi_pq_allstructuresHP[ph_HP_alternative][s] if s in phi_vs_phi_pq_allstructuresHP[ph_HP_alternative] else 0 for s in structure_listHP if s != ph_HP_alternative]
   CMO_list_for_plot_HP  = [CMO(paramHP.contact_map_list[s - 1], paramHP.contact_map_list[ph_HP_alternative - 1]) for s in structure_listHP if s != ph_HP_alternative]
   axindex = tuple((plotindex//5, plotindex%5))
   ax[axindex].scatter(CMO_list_for_plot_HP , phipq_list_HP , c='grey', s=3, alpha=0.3)
   ax[axindex].set_ylabel(r'$\phi_{qp}$')#+ '\nfor' + r' $p=$ '+ str(ph_RNA))
   ax[axindex].set_xlabel(r'CMO'+ '\n') #for' + r' $p=$ '+ str(ph_RNA))
   ax[axindex].set_title(ph_description, fontsize='small')
   ax[axindex].set_yscale('log')
   min_axlim = min([0.5,]+[phi_value for phi_value in phipq_list_HP if phi_value > 0])
   ax[axindex].set_ylim(0.5 * min_axlim, 1.2)
if nrows * 5 > len(examples_to_plot):
   for plotindex in range(len(examples_to_plot), nrows * 5):
      ax[plotindex//5, plotindex%5].axis('off')
f.tight_layout()
f.savefig('./plots/HP_CMO_many_examples'+str(paramHP.L)+'kbT' + str(int(10 *paramHP.kbT))+'.png', bbox_inches='tight', dpi=200)
plt.close('all')
del f, ax
## AIT
int_to_bin = {0: '00', 1: '10', 2: '11', 3: '01'}
f, ax = plt.subplots(nrows=nrows, ncols=5, figsize=(11, 1.7 * nrows + 1.7))
for plotindex, ph_HP_alternative in enumerate(examples_to_plot):
   ph_description = ''.join([int_to_bin[int(i)] for i in paramHP.contact_map_vs_updown[paramHP.contact_map_list[ph_HP_alternative -1]]])
   ph_description_title = ''.join([int_to_direction[int(i)] for i in paramHP.contact_map_vs_updown[paramHP.contact_map_list[ph_HP_alternative -1]]])
   phipq_list_HP = [phi_vs_phi_pq_allstructuresHP[ph_HP_alternative][s] if s in phi_vs_phi_pq_allstructuresHP[ph_HP_alternative] else 0 for s in structure_listHP if s != ph_HP_alternative]
   updown_list = [''.join([int_to_bin[int(i)] for i in paramHP.contact_map_vs_updown[paramHP.contact_map_list[s -1]]]) for s in structure_listHP if s != ph_HP_alternative]
   Dingle_pred_list = [calc_KC(s+ph_description) - calc_KC(ph_description) for s in updown_list]
   axindex = tuple((plotindex//5, plotindex%5))
   ax[axindex].scatter(Dingle_pred_list , phipq_list_HP , c='grey', s=3, alpha=0.3)
   ax[axindex].set_ylabel(r'$\phi_{qp}$')#+ '\nfor' + r' $p=$ '+ str(ph_RNA))
   ax[axindex].set_xlabel('AIT approach'+ '\n') #for' + r' $p=$ '+ str(ph_RNA))
   ax[axindex].set_title(ph_description_title, fontsize='small')
   ax[axindex].set_yscale('log')
   min_axlim = min([0.5,]+[phi_value for phi_value in phipq_list_HP if phi_value > 0])
   ax[axindex].set_ylim(0.5 * min_axlim, 1.2)
if nrows * 5 > len(examples_to_plot):
   for plotindex in range(len(examples_to_plot), nrows * 5):
      ax[plotindex//5, plotindex%5].axis('off')
f.tight_layout()
f.savefig('./plots/HP_Dingle_many_examples'+str(paramHP.L)+'kbT' + str(int(10 *paramHP.kbT))+'.png', bbox_inches='tight', dpi=200)
plt.close('all')
del f, ax
## exposed residues
f, ax = plt.subplots(nrows=nrows, ncols=5, figsize=(11, 1.7 * nrows + 1.7))
for plotindex, ph_HP_alternative in enumerate(examples_to_plot):
   ph_description_title = ''.join([int_to_direction[int(i)] for i in paramHP.contact_map_vs_updown[paramHP.contact_map_list[ph_HP_alternative -1]]])
   phipq_list_HP = [phi_vs_phi_pq_allstructuresHP[ph_HP_alternative][s] if s in phi_vs_phi_pq_allstructuresHP[ph_HP_alternative] else 0 for s in structure_listHP if s != ph_HP_alternative]
   surface_pattern_initial = pattern_internal_surface(paramHP.contact_map_list[ph_HP_alternative -1], paramHP.L)
   surface_pattern_list = [pattern_internal_surface(paramHP.contact_map_list[s -1], paramHP.L) for s in structure_listHP if s != ph_HP_alternative]
   surface_similarity_list = [len([x for i, x in enumerate(s) if x == surface_pattern_initial[i]])/len(s) for s in surface_pattern_list]
   axindex = tuple((plotindex//5, plotindex%5))
   ax[axindex].scatter(surface_similarity_list , phipq_list_HP , c='grey', s=3, alpha=0.3)
   ax[axindex].set_ylabel(r'$\phi_{qp}$')#+ '\nfor' + r' $p=$ '+ str(ph_RNA))
   ax[axindex].set_xlabel('similarity of\nsurface/core pattern'+ '\n') #for' + r' $p=$ '+ str(ph_RNA))
   ax[axindex].set_title(ph_description_title, fontsize='small')
   ax[axindex].set_yscale('log')
   min_axlim = min([0.5,]+[phi_value for phi_value in phipq_list_HP if phi_value > 0])
   ax[axindex].set_ylim(0.5 * min_axlim, 1.2)
if nrows * 5 > len(examples_to_plot):
   for plotindex in range(len(examples_to_plot), nrows * 5):
      ax[plotindex//5, plotindex%5].axis('off')
f.tight_layout()
f.savefig('./plots/HP_exposed_res_many_examples'+str(paramHP.L)+'kbT' + str(int(10 *paramHP.kbT))+'.png', bbox_inches='tight', dpi=200)
plt.close('all')
del f, ax
###############################################################################################
###############################################################################################
print( 'plot neutral set size data', flush=True)
###############################################################################################
###############################################################################################
for mfe_included in [True, False]:
  if mfe_included:
     type_Boltzmann_average = '' #['', 'no_mfe', 'folding_seq_only']
  else:
    type_Boltzmann_average = 'no_mfe'
  meanBoltzmann_all_filenameHP = './GPmapdata_HP/HPmeanBoltzmann_gsamplet_L'+str(paramHP.L)+type_Boltzmann_average+'_100kbT'+str(int(100*paramHP.kbT))  +'_'+str(paramHP.g_sample_size_meanBoltz_random_seq)+'.csv'
  df_Boltzmann_freq_HP = pd.read_csv(meanBoltzmann_all_filenameHP)  
  structure_vs_mean_Boltz_HP = {row['structure']: row['mean Boltzmann freq'] for rowindex, row in df_Boltzmann_freq_HP.iterrows()}
  #################
  if mfe_included:
     columname = 'Boltzmannmean' 
  else:
    columname = 'Boltzmannmean_nomfe' 
  #################
  f, ax = plt.subplots(ncols = 2, figsize=(8, 3.3))
  for i in range(2):
    ax[i].set_xscale('log')
    ax[i].set_yscale('log')
    ax[i].set_xlabel(r'phenotypic frequency $f_q$'+'\nof structure')
    if mfe_included:
      ax[i].set_ylabel(r'mean Boltzmann frequency $p_q$'+'\nin sequence sample')
    else:
      ax[i].set_ylabel(r'mean Boltzmann frequency $p_q$'+'\nas suboptimal structure\nin sequence sample')
  xRNA, yRNA = np.divide(df_NNE_dataRNA['Nmfe'].tolist(), 4.0**paramRNA.L_sampling), np.divide(df_NNE_dataRNA[columname].tolist(), 4.0**paramRNA.L_sampling)
  ax[0].scatter(xRNA, yRNA, s=15, c='grey', alpha=0.95)
  assert len(xRNA) == 50
  print('number of zero values in RNA neutral set size plot', len([x for x in yRNA if x == 0] + [x for x in xRNA if x == 0]))
  minx = min(min([x for x in xRNA if x > 0]), min([x for x in yRNA if x > 0]))
  maxx = max(max([x for x in xRNA if x > 0]), max([x for x in yRNA if x > 0]))
  ax[0].set_xlim(minx * 0.5, maxx *2)
  ax[0].set_ylim(minx * 0.5, maxx *2)
  ax[0].set_title('RNA secondary structure')
  xHP, yHP = [structure_vs_freq_estimateHP[s] for s in df_N_HP['structure'].tolist()[:]], [structure_vs_mean_Boltz_HP[s] for s in df_N_HP['structure'].tolist()[:]]
  ax[1].scatter(xHP, yHP, s=5, c='grey', alpha=0.7)
  minxfreq = min(xHP)
  maxxfreq = max(xHP)
  #minxBoltz, maxxBoltz = min([structure_vs_mean_Boltz_HP[s] for s in df_N_HP['structure'].tolist()[:]]), max([structure_vs_mean_Boltz_HP[s] for s in df_N_HP['structure'].tolist()[:]])
  ax[1].set_xlim(minxfreq * 0.5, maxxfreq *2)
  print('number of zero values in HP neutral set size plot', len([x for x in xHP if x == 0] + [x for x in yHP if x == 0]), 'out of', len(xHP))
  #ax[1].set_ylim(0.75 * 10**np.floor(np.log10(minxBoltz)), 10**np.ceil(np.log10(maxxBoltz)) *2)
  ax[1].set_title('HP protein model')
  f.tight_layout()
  f.savefig('./plots/HP_and_RNA_neutral_set_size_mean_Boltz'+type_Boltzmann_average+'.png', bbox_inches='tight')

###############################################################################################
###############################################################################################
print( 'plot data - sequence diversity for HP', flush=True)
###############################################################################################
###############################################################################################
from scipy.special import comb
df_HammingHP = pd.read_csv(paramHP.Hammingdist_neutral_set_filename)  
ph_vs_Hamming_dist_vs_freq = load_df_into_dict_of_dict(df_HammingHP, 'structure', 'Hamming dist', 'frequency', structure_listHP)
examples_to_plot = [structure_listHP[i] for i in range(0, len(structure_listHP), int(np.ceil(len(structure_listHP)/35)))]
nrows = int(np.ceil(len(examples_to_plot)/5))
K = 2
int_to_direction = {0: 'r', 1: 'l', 2: 'u', 3: 'd'}
#####
f, ax = plt.subplots(nrows=nrows, ncols=5, figsize=(11, 1.7 * nrows + 1.7))
for plotindex, ph_HP_alternative in enumerate(examples_to_plot):
   axindex = tuple((plotindex//5, plotindex%5))
   ph_description = ''.join([int_to_direction[int(i)] for i in paramHP.contact_map_vs_updown[paramHP.contact_map_list[ph_HP_alternative -1]]])
   dist_list =[k for k in ph_vs_Hamming_dist_vs_freq[ph_HP_alternative].keys()]
   dist_distribution = [ph_vs_Hamming_dist_vs_freq[ph_HP_alternative][k]/ sum(list(ph_vs_Hamming_dist_vs_freq[ph_HP_alternative].values())) for k in dist_list]
   ax[axindex].bar(dist_list, dist_distribution)
   ax[axindex].plot(list(range(paramHP.L + 1)), [comb(paramHP.L, H)/float(K**paramHP.L) for H in range(paramHP.L + 1)], c='k')
   ax[axindex].set_xlabel('Hamming dist. between\ntwo neutral set seq.')
   ax[axindex].set_ylabel('frequency')
   ax[axindex].set_title(ph_description)
   print('mean Hamming', sum([a *b for a, b in zip(ph_vs_Hamming_dist_vs_freq[ph_HP_alternative].keys(), dist_distribution)]))
   print(sorted([k for k in ph_vs_Hamming_dist_vs_freq[ph_HP_alternative].keys()]))
   print([ph_vs_Hamming_dist_vs_freq[ph_HP_alternative][x] for x in sorted([k for k in ph_vs_Hamming_dist_vs_freq[ph_HP_alternative].keys()])])
   print('\n\n')
f.tight_layout()
f.savefig('./plots/Hamming_dist_HP.png', bbox_inches='tight', dpi=200)
plt.close('all')
###############################################################################################
###############################################################################################
print( 'plot data - sequence diversity for RNA', flush=True)
###############################################################################################
###############################################################################################
df_HammingRNA = pd.read_csv(paramRNA.Hammingdist_neutral_set_filename)  
ph_vs_Hamming_dist_vs_freq = load_df_into_dict_of_dict(df_HammingRNA, 'structure', 'Hamming dist', 'frequency', structure_list_seq_sample)
examples_to_plot = [structure_list_seq_sample[i] for i in range(0, len(structure_list_seq_sample), int(np.ceil(len(structure_list_seq_sample)/35)))]
nrows = int(np.ceil(len(examples_to_plot)/5))
K = 4
#####
f, ax = plt.subplots(nrows=nrows, ncols=5, figsize=(11, 1.7 * nrows + 1.7))
for plotindex, ph_RNA_alternative in enumerate(examples_to_plot):
   axindex = tuple((plotindex//5, plotindex%5))
   Hamming_dist_range = [H - 0.5 for H in range(paramRNA.L_sampling + 2)]
   dist_list =[k for k in ph_vs_Hamming_dist_vs_freq[ph_RNA_alternative].keys()]
   dist_distribution = [ph_vs_Hamming_dist_vs_freq[ph_RNA_alternative][k]/ sum(list(ph_vs_Hamming_dist_vs_freq[ph_RNA_alternative].values())) for k in dist_list]
   ax[axindex].bar(dist_list, dist_distribution)
   ax[axindex].plot(list(range(paramRNA.L_sampling + 1)), [comb(paramRNA.L_sampling, H)*(K-1)**H/float(K**paramRNA.L_sampling) for H in range(paramRNA.L_sampling + 1)], c='k')
   ax[axindex].set_xlabel('Hamming dist. between\ntwo neutral set seq.')
   ax[axindex].set_ylabel('frequency')
   ax[axindex].set_title(ph_RNA_alternative)
   print('mean Hamming', sum([a *b for a, b in zip(ph_vs_Hamming_dist_vs_freq[ph_RNA_alternative].keys(), dist_distribution)]))
f.tight_layout()
f.savefig('./plots/Hamming_dist_RNA.png', bbox_inches='tight', dpi=200)
plt.close('all')




