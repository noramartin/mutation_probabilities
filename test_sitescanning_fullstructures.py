#!/usr/bin/env python3
import numpy as np
from functools import partial
import pandas as pd
from functions_fullstructures.Nestimates_gsampling import g_sampling_seq_sample_mfe
from multiprocessing import Pool
from os.path import isfile
import parameters_fullstructures as param
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import functions_fullstructures.functions_for_sampling_with_bp_swap as sample
from functions_fullstructures.general_functions import  get_phipq_sequence_sample, get_boltzmann_sequence_sample, load_df_into_dict_of_dict
import functions_fullstructures.thermodynamic_functions as DG
from copy import deepcopy


def overlap_two_lists(l1, l2):
   distinct_elements = len(list(set(l1 + l2)))
   return 1 - distinct_elements/(len(list(set(l1))) + len(list(set(l2)))) * 2


###################################################################################################
seq_per_str = 10**4 #10**3
cutoff_phipq = 4.9/(seq_per_str * 3 * param.L_sampling)
cutoff_Boltz = 0.2/seq_per_str
###############################################################################################
print( 'test site-scanning')
###############################################################################################
if not (isfile(param.meanBoltzmann_neutral_set_filename_gsampling) and isfile(param.phipq_filename_gsampling)): 
   structure_sample = sample.read_structure_list(param.filename_structure_list)
   ###############################################################################################
   print( 'g-sample' )
   ###############################################################################################
   structure_vs_seq_sample = g_sampling_seq_sample_mfe(structure_sample, param.sample_size_gsample * 10,  
                                                      max_sample_at_once= 10**4, seq_per_structure = seq_per_str)
   for data in ['phi', 'Boltzmann']:
      structure_sample_gsampling = [s for s in structure_sample if len(structure_vs_seq_sample[s])//2 > 100]
      print('fraction of overlap in samples', [overlap_two_lists(structure_vs_seq_sample[s][:len(structure_vs_seq_sample[s])//2], structure_vs_seq_sample[s][len(structure_vs_seq_sample[s])//2:]) for s in structure_sample_gsampling])
      if data == 'phi':
         ###############################################################################################
         ###############################################################################################
         print(  'get phipq stats for each sequence sample')
         ###############################################################################################
         ###############################################################################################
         function_for_parallelised_calc = partial(get_phipq_sequence_sample, 
                                                  folding_function=DG.get_unique_mfe_structure_seq_str, min_phi = cutoff_phipq)
         
         ###
         seq_sample = [structure_vs_seq_sample[s][:len(structure_vs_seq_sample[s])//2] for s in structure_sample_gsampling]
         with Pool(param.ncpus) as p:
            pool_result = p.map(function_for_parallelised_calc, seq_sample)
         phi_vs_phi_pq_allstructures = {structure: deepcopy(pool_result[structureindex]) for structureindex, structure in enumerate(structure_sample_gsampling) } 
      if data == 'Boltzmann':
         ###############################################################################################
         ###############################################################################################
         print(  'get stats for each sequence sample')
         ###############################################################################################
         ###############################################################################################
         function_for_parallelised_calc = partial(get_boltzmann_sequence_sample, 
                                                  boltzmann_ensemble_function=DG.get_Boltzmann_freq_dict_lowenergyrange, min_Boltz = cutoff_Boltz)
         seq_sample = [structure_vs_seq_sample[s][len(structure_vs_seq_sample[s])//2:] for s in structure_sample_gsampling]

         with Pool(param.ncpus) as p:
            pool_result = p.map(function_for_parallelised_calc, seq_sample)
         ph_vs_Boltzmann_freq_in_neutral_set_allstructures = {structure: deepcopy(pool_result[structureindex]) for structureindex, structure in enumerate(structure_sample_gsampling)} 
      del function_for_parallelised_calc

   ###############################################################################################
   ###############################################################################################
   print(  'save data')
   ###############################################################################################
   ###############################################################################################
   structure_one_list, structure_two_list, phi_pq_list = zip(*[(structureone, structuretwo, phi) for structureone in structure_sample_gsampling for structuretwo, phi in phi_vs_phi_pq_allstructures[structureone].items()])
   df_phipq = pd.DataFrame.from_dict({'structure 1': structure_one_list, 'structure 2': structure_two_list, 'phi': phi_pq_list})
   df_phipq.to_csv(param.phipq_filename_gsampling)
   structure_one_list_boltz, structure_two_list_boltz, bfreq_list = zip(*[(structureone, structuretwo, f) for structureone in structure_sample_gsampling for structuretwo, f in ph_vs_Boltzmann_freq_in_neutral_set_allstructures[structureone].items()])
   df_Boltzmann_freq = pd.DataFrame.from_dict({'structure 1': structure_one_list_boltz, 'structure 2': structure_two_list_boltz, 'Boltzmann frequency': bfreq_list})
   df_Boltzmann_freq.to_csv(param.meanBoltzmann_neutral_set_filename_gsampling)
###############################################################################################
###############################################################################################
print( 'load data -NNE for sorting structures', flush=True)
###############################################################################################
###############################################################################################
df_NNE_dataRNA = pd.read_csv(param.Nresult_filename)
structure_vs_freq_estimateNNE = {row['structure']: row['Nmfe'] for rowindex, row in df_NNE_dataRNA.iterrows()}
structure_sample = sample.read_structure_list(param.filename_structure_list)
###############################################################################################
###############################################################################################
print( 'load data gsample', flush=True)
###############################################################################################
###############################################################################################
df_phipq2 = pd.read_csv(param.phipq_filename_gsampling)
structure_list_gsample = sorted(list(set(df_phipq2['structure 1'].tolist()[:])), key=structure_vs_freq_estimateNNE.get, reverse=True) #only structures with g-sampling results
df_Boltzmann_freq2 = pd.read_csv(param.meanBoltzmann_neutral_set_filename_gsampling)  
ph_vs_Boltzmann_freq_in_neutral_set_allstructuresRNA2 = load_df_into_dict_of_dict(df_Boltzmann_freq2, 'structure 1', 'structure 2', 'Boltzmann frequency', structure_sample)
phi_vs_phi_pq_allstructuresRNA2 = load_df_into_dict_of_dict(df_phipq2, 'structure 1', 'structure 2', 'phi', structure_sample)

###############################################################################################
###############################################################################################
print( 'load data', flush=True)
###############################################################################################
###############################################################################################
df_phipq = pd.read_csv(param.phipq_filename_sampling)
df_Boltzmann_freq = pd.read_csv(param.meanBoltzmann_neutral_set_filename_sampling)  
ph_vs_Boltzmann_freq_in_neutral_set_allstructuresRNA = load_df_into_dict_of_dict(df_Boltzmann_freq, 'structure 1', 'structure 2', 'Boltzmann frequency', structure_sample)
phi_vs_phi_pq_allstructuresRNA = load_df_into_dict_of_dict(df_phipq, 'structure 1', 'structure 2', 'phi', structure_sample)
###############################################################################################
###############################################################################################
print( 'plot data - further examples for RNA', flush=True)
###############################################################################################
###############################################################################################
examples_to_plot = [structure_list_gsample[i] for i in range(0, len(structure_list_gsample), int(np.ceil(len(structure_list_gsample)/35)))]
nrows = int(np.ceil(len(examples_to_plot)/5))
print('plotting', len(examples_to_plot), 'out of', len(structure_list_gsample), 'phenotypes - in ', nrows, 'rows')
f, ax = plt.subplots(nrows=nrows, ncols=5, figsize=(11, 1.7 * nrows + 1.7))
for plotindex, ph_RNA_alternative in enumerate(examples_to_plot):
   structure_list_all = list(set([s for s in ph_vs_Boltzmann_freq_in_neutral_set_allstructuresRNA[ph_RNA_alternative] if s not in ['|', '.']] + [s for s in ph_vs_Boltzmann_freq_in_neutral_set_allstructuresRNA2[ph_RNA_alternative] if s not in ['|', '.']]))
   boltz_freq_list_RNA = [ph_vs_Boltzmann_freq_in_neutral_set_allstructuresRNA[ph_RNA_alternative][s] if s in ph_vs_Boltzmann_freq_in_neutral_set_allstructuresRNA[ph_RNA_alternative] else 0 for s in structure_list_all if s != ph_RNA_alternative]
   boltz_freq_list_RNA2 = [ph_vs_Boltzmann_freq_in_neutral_set_allstructuresRNA2[ph_RNA_alternative][s] if s in ph_vs_Boltzmann_freq_in_neutral_set_allstructuresRNA2[ph_RNA_alternative] else 0 for s in structure_list_all if s != ph_RNA_alternative]
   if nrows == 1:
      axindex = plotindex
   else:
      axindex = tuple((plotindex//5, plotindex%5))
   ax[axindex].scatter(boltz_freq_list_RNA, boltz_freq_list_RNA2, c='grey', s=3, alpha=0.4)
   ax[axindex].set_ylabel(r'$p_{qp}$'+'\n(sequence-sampling)')#+ '\nfor' + r' $p=$ '+ str(ph_RNA))
   ax[axindex].set_xlabel(r'$p_{qp}$'+'\n(site-scanning)') #for' + r' $p=$ '+ str(ph_RNA))
   ax[axindex].set_title(ph_RNA_alternative, fontsize='small')
   ax[axindex].set_xscale('log')
   ax[axindex].set_yscale('log')
   min_axlim = min([0.5,]+[min(g, phi_value) for (g, phi_value) in zip(boltz_freq_list_RNA, boltz_freq_list_RNA2) if g > 0 and phi_value > 0])
   ax[axindex].plot([min_axlim * 10.0**(-7), 2], [min_axlim * 10.0**(-7), 2], c='k', lw=0.5)
   ax[axindex].set_ylim(0.5 * min_axlim, 1.2)
   ax[axindex].set_xlim(0.5 * min_axlim, 1.2)
if nrows * 5 > len(examples_to_plot):
   for plotindex in range(len(examples_to_plot), nrows * 5):
      ax[plotindex//5, plotindex%5].axis('off')
f.tight_layout()
f.savefig('./plots/RNA_Boltz_many_examples_test'+str(param.sample_filename)+'.png', bbox_inches='tight', dpi=200)
plt.close('all')
del f, ax
f, ax = plt.subplots(nrows=nrows, ncols=5, figsize=(11, 1.7 * nrows + 1.7))
for plotindex, ph_RNA_alternative in enumerate(examples_to_plot):
   structure_list_all = list(set([s for s in phi_vs_phi_pq_allstructuresRNA[ph_RNA_alternative] if s != '|'] + [s for s in phi_vs_phi_pq_allstructuresRNA2[ph_RNA_alternative] if s != '|']))
   phipq_list_RNA = [phi_vs_phi_pq_allstructuresRNA[ph_RNA_alternative][s] if s in phi_vs_phi_pq_allstructuresRNA[ph_RNA_alternative] else 0 for s in structure_list_all if s != ph_RNA_alternative]
   phipq_list_RNA2 = [phi_vs_phi_pq_allstructuresRNA2[ph_RNA_alternative][s] if s in phi_vs_phi_pq_allstructuresRNA2[ph_RNA_alternative] else 0 for s in structure_list_all if s != ph_RNA_alternative]
   if nrows == 1:
      axindex = plotindex
   else:
      axindex = tuple((plotindex//5, plotindex%5))
   ax[axindex].scatter(phipq_list_RNA, phipq_list_RNA2, c='grey', s=3, alpha=0.3)
   ax[axindex].set_xlabel(r'$\phi_{qp}$'+'\n(site-scanning)')#+ '\nfor' + r' $p=$ '+ str(ph_RNA))
   ax[axindex].set_ylabel(r'$\phi_{qp}$'+'\n(sequence-sampling)') #for' + r' $p=$ '+ str(ph_RNA))
   ax[axindex].set_title(ph_RNA_alternative, fontsize='small')
   ax[axindex].set_xscale('log')
   ax[axindex].set_yscale('log')
   min_axlim = min([0.5,]+[min(g, phi_value) for (g, phi_value) in zip(phipq_list_RNA2, phipq_list_RNA) if g > 0 and phi_value > 0])
   ax[axindex].plot([min_axlim * 10.0**(-7), 2], [min_axlim * 10.0**(-7), 2], c='k', lw=0.5)
   ax[axindex].set_ylim(0.5 * min_axlim, 1.2)
   ax[axindex].set_xlim(0.5 * min_axlim, 1.2)
if nrows * 5 > len(examples_to_plot):
   for plotindex in range(len(examples_to_plot), nrows * 5):
      ax[plotindex//5, plotindex%5].axis('off')
f.tight_layout()
f.savefig('./plots/RNA_freq_many_examples_test'+str(param.sample_filename)+'.png', bbox_inches='tight', dpi=200)
plt.close('all')
del f, ax

