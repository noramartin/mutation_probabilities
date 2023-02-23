#!/usr/bin/env python3
import numpy as np
import pandas as pd
from os.path import isfile
import parameters_HP as param
from functions_fullstructures.general_functions import logPearsonR_nonzero_values, correct_top_x_predicted, load_df_into_dict_of_dict, linlogPearsonR_nonzero_values, Hamming_dist
from scipy.stats import spearmanr
from HP_model.HPfunctions import *
from functools import partial
from collections import Counter



def find_x_sequences_per_structure(GPmap, number_sequences_per_structure, structure_list):
   """returns number_sequences_per_structure structures drawn for each strucutre that is present in the GPmap 
   and passes the structure_invalid_test function with False;
   sampling is performed with replacement"""
   structure_vs_seq = {structure_int: [] for structure_int in structure_list}
   for structure_int in structure_list:
      seqlist_NN_array = np.argwhere(GPmap==structure_int)
      for seq_index in np.random.choice(len(seqlist_NN_array), number_sequences_per_structure, replace=True):
         structure_vs_seq[structure_int].append(tuple([t for t in seqlist_NN_array[seq_index]]))
         assert GPmap[tuple(seqlist_NN_array[seq_index])] == structure_int
   return structure_vs_seq

def neighbours_gHP(g, L): 
   """list all pont mutational neighbours of sequence g (integer notation)"""
   return [tuple([oldK if gpos!=pos else new_K for gpos, oldK in enumerate(g)]) for pos in range(L) for new_K in [1, 0] if g[pos]!=new_K]



def is_unfolded_HP_structure(s):
   return s < 0.5

def get_frequencies_in_array(GPmap, ignore_undefined=True, structure_invalid_test=is_unfolded_HP_structure):
   """ sum over an entire GP array: and sum the number of times each structure is found;
   if ignore_undefined=True, the function structure_invalid_test will determine, which structures are selected; 
   otherwise all structures will be selected"""
   ph_vs_f = {}
   for p in GPmap.copy().flat:
      try:
         ph_vs_f[p] += 1
      except KeyError:
         ph_vs_f[p] = 1
   if not ignore_undefined:
      return ph_vs_f
   else:
      return {p:f for p, f in ph_vs_f.items() if not structure_invalid_test(p)}


def phi_pq_function(GPmap, ph_vs_f, structure_invalid_test=is_unfolded_HP_structure):
   print( 'mutational neighbourhood - rh o and phipq')
   L, K = GPmap.ndim, GPmap.shape[0]
   ph_vs_phi_pq = {ph: {} for ph in ph_vs_f}
   for genotype, ph in np.ndenumerate(GPmap):
      if not structure_invalid_test(ph):
         neighbours = neighbours_gHP(tuple(genotype), L)
         for neighbourgeno in neighbours:
            neighbourpheno = GPmap[neighbourgeno]
            try:
               ph_vs_phi_pq[ph][neighbourpheno] += 1.0/(ph_vs_f[ph] * (K - 1) * L)
            except KeyError:
               ph_vs_phi_pq[ph][neighbourpheno] = 1.0/(ph_vs_f[ph] * (K - 1) * L)
   return ph_vs_phi_pq

plot_further_stats = False
###############################################################################################
###############################################################################################
GPmap = np.load(param.GPmap_filename)
###############################################################################################
###############################################################################################
print( 'neutral sets and phipq', flush=True)
###############################################################################################
###############################################################################################
if not (isfile(param.neutral_set_size_filename) and isfile(param.phipq_filename) and isfile(param.meanBoltzmann_neutral_set_filename) and isfile(param.Hammingdist_neutral_set_filename)):
   structure_vs_neutral_set_size = get_frequencies_in_array(GPmap, structure_invalid_test=is_unfolded_HP_structure)
   structure_list = sorted([s for s in structure_vs_neutral_set_size], key=structure_vs_neutral_set_size.get, reverse=True)
   df_N = pd.DataFrame.from_dict({'structure': structure_list, 'neutral set size': [structure_vs_neutral_set_size[s] for s in structure_list]})
   df_N.to_csv(param.neutral_set_size_filename)
   structure_vs_freq_estimate = {s: N/float(param.K**param.L) for s, N in structure_vs_neutral_set_size.items()}
   #######
   phi_vs_phi_pq_allstructures = phi_pq_function(GPmap, structure_vs_neutral_set_size)
   structure_one_list, structure_two_list, phi_pq_list = zip(*[(structureone, structuretwo, phi) for structureone in structure_list for structuretwo, phi in phi_vs_phi_pq_allstructures[structureone].items()])
   df_phipq = pd.DataFrame.from_dict({'structure 1': structure_one_list, 'structure 2': structure_two_list, 'phi': phi_pq_list})
   df_phipq.to_csv(param.phipq_filename)
   #####
   print('Boltzmann stats')
   #####
   structure_vs_seq = find_x_sequences_per_structure(GPmap, param.number_sequences_per_structure, [s for s in structure_vs_freq_estimate.keys()])
   ph_vs_Boltzmann_freq_in_neutral_set_allstructures = {structure_int: {} for structure_int in structure_list}
   for structure1 in structure_list:
      seq_vs_boltz_list = {seq: HPget_Boltzmann_freq_list(seq, param.contact_map_list, kbT = param.kbT) for seq in structure_vs_seq[structure1]}
      for structure2index, structure2 in enumerate(structure_list):
         meanboltz = np.mean([boltz_list[structure2 - 1] for boltz_list in seq_vs_boltz_list.values()])
         ph_vs_Boltzmann_freq_in_neutral_set_allstructures[structure1][structure2] = meanboltz
   structure_one_list_boltz, structure_two_list_boltz, bfreq_list = zip(*[(structureone, structuretwo, f) for structureone in structure_list for structuretwo, f in ph_vs_Boltzmann_freq_in_neutral_set_allstructures[structureone].items()])
   df_Boltzmann_freq = pd.DataFrame.from_dict({'structure 1': structure_one_list_boltz, 'structure 2': structure_two_list_boltz, 'Boltzmann frequency': bfreq_list})
   df_Boltzmann_freq.to_csv(param.meanBoltzmann_neutral_set_filename)
   #####
   print('Hamming dist distribution')
   #####
   structure_vs_seq = find_x_sequences_per_structure(GPmap, param.number_sequences_per_structure, structure_list)
   structure_vs_dist_list = {s: Counter([Hamming_dist(seq1, seq2) for seqindex, seq1 in enumerate(seq_list) for seq2 in seq_list[:seqindex]]) for s, seq_list in structure_vs_seq.items()}
   structure_list_hamming, hamming_list, hfreq_list = zip(*[(structureone, dist, f) for structureone in structure_list for dist, f in structure_vs_dist_list[structureone].items()])
   
   df_Hamming = pd.DataFrame.from_dict({'structure': structure_list_hamming, 'Hamming dist': hamming_list, 'frequency': hfreq_list})
   df_Hamming.to_csv(param.Hammingdist_neutral_set_filename)

else:
   df_N = pd.read_csv(param.neutral_set_size_filename)
   structure_vs_freq_estimate = {row['structure']: row['neutral set size']/float(param.K**param.L) for rowindex, row in df_N.iterrows()}
   structure_list = df_N['structure'].tolist()[:]
   df_phipq = pd.read_csv(param.phipq_filename)
   phi_vs_phi_pq_allstructures = load_df_into_dict_of_dict(df_phipq, 'structure 1', 'structure 2', 'phi', structure_list)
   df_Boltzmann_freq = pd.read_csv(param.meanBoltzmann_neutral_set_filename)  
   ph_vs_Boltzmann_freq_in_neutral_set_allstructures = load_df_into_dict_of_dict(df_Boltzmann_freq, 'structure 1', 'structure 2', 'Boltzmann frequency', structure_list)
###############################################################################################
###############################################################################################
################################################################################################
###############################################################################################
print( 'get data on correlations', flush=True )
###############################################################################################
###############################################################################################
int_to_bin = {0: '00', 1: '10', 2: '11', 3: '01'}
from KC import calc_KC

for type_correlation, correlation_function in [('Pearson', logPearsonR_nonzero_values), ('Spearman', spearmanr), ('top-10', partial(correct_top_x_predicted, x=10)), ('top-20', partial(correct_top_x_predicted, x=20)), ('top-30', partial(correct_top_x_predicted, x=30))]:
   corr_Boltz_phipqS, corr_freq_phipqS, structure_list_for_df, corr_CMO_phipqS, corr_AIT_phipqS, corr_surface_phipqS = [], [], [], [], [], []
   for ph in structure_list:  
      ###
      boltz_freq_list = [ph_vs_Boltzmann_freq_in_neutral_set_allstructures[ph][s] for s in structure_list if s != ph]
      freq_list = [structure_vs_freq_estimate[s] for s in structure_list if s != ph]
      phipq_list_s = [phi_vs_phi_pq_allstructures[ph][s] if s in phi_vs_phi_pq_allstructures[ph] else 0 for s in structure_list if s != ph]
      ###
      CMO_list_for_plot_HP  = [CMO(param.contact_map_list[s - 1], param.contact_map_list[ph - 1]) for s in structure_list if s != ph]
      ph_description = ''.join([int_to_bin[int(i)] for i in param.contact_map_vs_updown[param.contact_map_list[ph -1]]])
      updown_list = [''.join([int_to_bin[int(i)] for i in param.contact_map_vs_updown[param.contact_map_list[s -1]]]) for s in structure_list if s != ph]
      Dingle_pred_list = [calc_KC(s+ph_description) - calc_KC(ph_description) for s in updown_list]
      #
      surface_pattern_initial = pattern_internal_surface(param.contact_map_list[ph -1], param.L)
      surface_pattern_list = [pattern_internal_surface(param.contact_map_list[s -1], param.L) for s in structure_list if s != ph]
      surface_sim_list = [len([x for i, x in enumerate(s) if x == surface_pattern_initial[i]])/len(s) for s in surface_pattern_list]
      try:
         corr_Boltz_phipqS.append(correlation_function(boltz_freq_list, phipq_list_s)[0])
         corr_freq_phipqS.append(correlation_function(freq_list, phipq_list_s)[0])
         structure_list_for_df.append(ph)
         if type_correlation == 'Pearson':
            corr_CMO_phipqS.append(linlogPearsonR_nonzero_values(CMO_list_for_plot_HP, phipq_list_s)[0])
            corr_surface_phipqS.append(linlogPearsonR_nonzero_values(surface_sim_list, phipq_list_s)[0])
            corr_AIT_phipqS.append(linlogPearsonR_nonzero_values([max(Dingle_pred_list) + 1 - d for d in Dingle_pred_list], phipq_list_s)[0])
         else:
            corr_CMO_phipqS.append(correlation_function(CMO_list_for_plot_HP, phipq_list_s)[0])
            corr_surface_phipqS.append(correlation_function(surface_sim_list, phipq_list_s)[0])
            corr_AIT_phipqS.append(correlation_function([max(Dingle_pred_list) + 1 - d for d in Dingle_pred_list], phipq_list_s)[0])
      except ValueError: #ignore if two or less non-zero values
         corr_Boltz_phipqS.append(np.nan)
         corr_freq_phipqS.append(np.nan)
         structure_list_for_df.append(ph)
         corr_CMO_phipqS.append(np.nan)
         corr_AIT_phipqS.append(np.nan)
      del boltz_freq_list, freq_list, phipq_list_s
   df_corr =  pd.DataFrame.from_dict({'structure': structure_list_for_df, 
                                      type_correlation + ' correlation mean Boltzmann frequ. vs phiqp substitutions': corr_Boltz_phipqS,
                                      type_correlation + ' correlation structure frequ. vs phiqp substitutions': corr_freq_phipqS,
                                      type_correlation + ' correlation CMO vs phiqp substitutions': corr_CMO_phipqS,
                                      type_correlation + ' correlation (1 - conditional complexity) vs phiqp substitutions': corr_AIT_phipqS,
                                      type_correlation + ' correlation overlap of core positions vs phiqp substitutions': corr_surface_phipqS,
                                      'phenotypic frequency': [structure_vs_freq_estimate[ph] for ph in structure_list_for_df]})
   df_corr.to_csv('./GPmapdata_HP/Psampling_phenotype_phipq_correlations'+type_correlation+'_'+str(param.L)+'kbT' + str(param.kbT)+'.csv')
