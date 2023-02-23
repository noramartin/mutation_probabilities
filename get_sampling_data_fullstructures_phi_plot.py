#!/usr/bin/env python3
import numpy as np
from copy import deepcopy
import pandas as pd
from os.path import isfile
import functions_fullstructures.functions_for_sampling_with_bp_swap as sample
import functions_fullstructures.thermodynamic_functions as DG
from functions_fullstructures.general_functions import load_df_into_dict_of_dict, logPearsonR_nonzero_values, SpearmanR_nonzero_values, correct_top_x_predicted, get_phipq_sequence_sample, get_boltzmann_sequence_sample, linlogPearsonR_nonzero_values, Hamming_dist
from multiprocessing import Pool
from functools import partial
import parameters_fullstructures as param
from functions_fullstructures.Nestimates_gsampling import g_sampling_Nestimate_mfe_allstructures, g_sampling_struct_sample_balance_n_stacks
from functions_fullstructures.rna_structural_functions import sequence_compatible_with_basepairs, get_basepair_indices_from_dotbracket, is_likely_to_be_valid_structure
import RNA
import random

print('sampling length', param.L_sampling)
print('number sequences per structure', param.number_walks*param.length_per_walk/param.every_Nth_seq)
seq_per_str = param.number_walks *  param.length_per_walk//param.every_Nth_seq
cutoff_phipq = 4.9/(seq_per_str * 3 * param.L_sampling)
cutoff_Boltz = 0.2/seq_per_str
number_measurements = 3
random.seed(7)

###############################################################################################
###############################################################################################
print(  'phipq and Boltzmann data')
###############################################################################################
###############################################################################################
samplename_vs_struct_vs_seq = {}
if not (isfile(param.meanBoltzmann_neutral_set_filename_sampling) and isfile(param.phipq_filename_sampling)):
   ###############################################################################################
   ###############################################################################################
   print(  'structure sample')
   ###############################################################################################
   ###############################################################################################
   if isfile(param.filename_structure_list):
      structure_sample = sample.read_structure_list(param.filename_structure_list)
   else:
      structure_sample = g_sampling_struct_sample_balance_n_stacks(param.L_sampling, param.num_sample_str, param.sample_size_gsample, max_sample_at_once= 10**4)
      #sample.generate_all_allowed_dotbracket(param.L_sampling, param.max_no_trials_RNAinverse, param.num_sample, 
      #                                                          allow_isolated_bps=False, test_RNAinverse=True)
      sample.save_structure_list(structure_sample, param.filename_structure_list)
   print( 'sample size:', len(structure_sample))
   for data in ['phi', 'Boltzmann']:
      filename_seq_sample = './GPmapdata_fullstructures/sequence_sample'+data +param.sample_filename+'mfe_unique.csv'
      if not isfile(filename_seq_sample):
         ###############################################################################################
         ###############################################################################################
         print(  'sequence sample')
         ###############################################################################################
         ###############################################################################################
         site_scanning_function = partial(sample.get_x_random_walks, 
                                      number_walks=param.number_walks, length_per_walk=param.length_per_walk, every_Nth_seq=param.every_Nth_seq)
         p = Pool(param.ncpus)
         with p:
            seq_list_each_structure_list = p.map(site_scanning_function, structure_sample)
         structure_sample_df = pd.DataFrame.from_dict({'structure': [structure_sample[i][:] for i, seq_list in enumerate(seq_list_each_structure_list) for seq in seq_list],
                                                       'seq': [seq for i, seq_list in enumerate(seq_list_each_structure_list) for seq in seq_list]})
         structure_sample_df.to_csv(filename_seq_sample)
         del site_scanning_function
      else:
         structure_sample_df = pd.read_csv(filename_seq_sample)
         samplename_vs_struct_vs_seq[data] = {structure: [] for structure in structure_sample}
         for i, row in structure_sample_df.iterrows(): 
            samplename_vs_struct_vs_seq[data][row['structure']].append(row['seq'])
         seq_list_each_structure_list = [[seq[:] for seq in samplename_vs_struct_vs_seq[data][s]] for s in structure_sample]
      print('total # sequences:', len([s  for slist in seq_list_each_structure_list for s in slist]))
      ### compare sequences
      if len(samplename_vs_struct_vs_seq.keys()) > 1:
         seq_Boltz_set = list(set([seq for l in samplename_vs_struct_vs_seq['Boltzmann'].values() for seq in l]))
         seq_phi_set = list(set([seq for l in samplename_vs_struct_vs_seq['phi'].values() for seq in l]))
         print('total number of distinct sequences in Boltzmann sample', len(seq_Boltz_set))
         print('total number of distinct sequences in phi sample', len(seq_phi_set))
         print('total number of distinct sequences in both samples', len(list(set(seq_Boltz_set + seq_phi_set))))

      if data == 'phi':
         ###############################################################################################
         ###############################################################################################
         print(  'get phipq stats for each sequence sample')
         ###############################################################################################
         ###############################################################################################
         function_for_parallelised_calc = partial(get_phipq_sequence_sample, 
                                                  folding_function=DG.get_unique_mfe_structure_seq_str, min_phi = cutoff_phipq)
         
         ###
         with Pool(param.ncpus) as p:
            pool_result = p.map(function_for_parallelised_calc, seq_list_each_structure_list)
         phi_vs_phi_pq_allstructures = {structure: deepcopy(pool_result[structureindex]) for structureindex, structure in enumerate(structure_sample) } 
      if data == 'Boltzmann':
         ###############################################################################################
         ###############################################################################################
         print(  'get stats for each sequence sample')
         ###############################################################################################
         ###############################################################################################
         function_for_parallelised_calc = partial(get_boltzmann_sequence_sample, 
                                                  boltzmann_ensemble_function=DG.get_Boltzmann_freq_dict_lowenergyrange, min_Boltz = cutoff_Boltz)
         with Pool(param.ncpus) as p:
            pool_result = p.map(function_for_parallelised_calc, seq_list_each_structure_list)
         ph_vs_Boltzmann_freq_in_neutral_set_allstructures = {structure: deepcopy(pool_result[structureindex]) for structureindex, structure in enumerate(structure_sample)} 
      del seq_list_each_structure_list, function_for_parallelised_calc

   ###############################################################################################
   ###############################################################################################
   print(  'save data')
   ###############################################################################################
   ###############################################################################################
   structure_one_list, structure_two_list, phi_pq_list = zip(*[(structureone, structuretwo, phi) for structureone in structure_sample for structuretwo, phi in phi_vs_phi_pq_allstructures[structureone].items()])
   df_phipq = pd.DataFrame.from_dict({'structure 1': structure_one_list, 'structure 2': structure_two_list, 'phi': phi_pq_list})
   df_phipq.to_csv(param.phipq_filename_sampling)
   structure_one_list_boltz, structure_two_list_boltz, bfreq_list = zip(*[(structureone, structuretwo, f) for structureone in structure_sample for structuretwo, f in ph_vs_Boltzmann_freq_in_neutral_set_allstructures[structureone].items()])
   df_Boltzmann_freq = pd.DataFrame.from_dict({'structure 1': structure_one_list_boltz, 'structure 2': structure_two_list_boltz, 'Boltzmann frequency': bfreq_list})
   df_Boltzmann_freq.to_csv(param.meanBoltzmann_neutral_set_filename_sampling)
###############################################################################################
###############################################################################################
print( 'get frequencies of all alternative phenotypes')
###############################################################################################
###############################################################################################
if not isfile(param.neutral_set_size_filename_sampling):# and param.L_sampling >= 22:
   structure_vs_freq = g_sampling_Nestimate_mfe_allstructures(param.L_sampling, param.sample_size_gsample,  max_sample_at_once= 10**4, min_count = 5)
   df_N = pd.DataFrame.from_dict({'structure': [s for s in structure_vs_freq], 'frequency': [structure_vs_freq[s] for s in structure_vs_freq]})
   df_N.to_csv(param.neutral_set_size_filename_sampling)
###############################################################################################
###############################################################################################
print( 'load data', flush=True)
###############################################################################################
###############################################################################################
df_N = pd.read_csv(param.neutral_set_size_filename_sampling)
structure_vs_freq_estimate = {row['structure']: row['frequency'] for rowindex, row in df_N.iterrows()}
df_phipq = pd.read_csv(param.phipq_filename_sampling)
df_NNE_dataRNA = pd.read_csv(param.Nresult_filename)
structure_vs_freq_estimateNNE = {row['structure']: row['Nmfe'] for rowindex, row in df_NNE_dataRNA.iterrows()}
structure_list_seq_sample = sorted(list(set(df_phipq['structure 1'].tolist()[:])), key=structure_vs_freq_estimateNNE.get, reverse=True)
df_Boltzmann_freq = pd.read_csv(param.meanBoltzmann_neutral_set_filename_sampling)  
ph_vs_Boltzmann_freq_in_neutral_set_allstructures = load_df_into_dict_of_dict(df_Boltzmann_freq, 'structure 1', 'structure 2', 'Boltzmann frequency', structure_list_seq_sample)
phi_vs_phi_pq_allstructures = load_df_into_dict_of_dict(df_phipq, 'structure 1', 'structure 2', 'phi', structure_list_seq_sample)
###############################################################################################
###############################################################################################
print(  'compatible set overlap', flush=True)
###############################################################################################
###############################################################################################
def fraction_compatible_seqlist(structure, seq_list):
   return len([seq for seq in seq_list if sequence_compatible_with_basepairs(seq, structure)])/len(seq_list)

if not isfile(param.compatible_filename_sampling):
   filename_seq_sample = './GPmapdata_fullstructures/sequence_sample'+'Boltzmann' +param.sample_filename+'mfe_unique.csv'
   structure_sample_df = pd.read_csv(filename_seq_sample)
   struct_vs_seq = {structure: [] for structure in phi_vs_phi_pq_allstructures}
   for i, row in structure_sample_df.iterrows(): 
      struct_vs_seq[row['structure']].append(row['seq'])
   phi_vs_compatible_seq_metric_allstructures = {p: {} for p in phi_vs_phi_pq_allstructures}
   for ph in phi_vs_compatible_seq_metric_allstructures:
      print('ph', ph, flush=True)
      pool_function = partial(fraction_compatible_seqlist, seq_list=deepcopy(struct_vs_seq[ph]))
      new_struct = [p for p in phi_vs_phi_pq_allstructures[ph].keys() if p!= '|']
      with Pool(25) as p:
         pool_result = p.map(pool_function, new_struct)
      for q, f in zip(new_struct, pool_result):
         phi_vs_compatible_seq_metric_allstructures[ph][q] = f
   structure_one_list, structure_two_list, seq_comp_list = zip(*[(structureone, structuretwo, phi) for structureone in phi_vs_compatible_seq_metric_allstructures for structuretwo, phi in phi_vs_compatible_seq_metric_allstructures[structureone].items()])
   df_comp = pd.DataFrame.from_dict({'structure 1': structure_one_list, 'structure 2': structure_two_list, 'fraction comp seq': seq_comp_list})
   df_comp.to_csv(param.compatible_filename_sampling)
   
## load 
df_comp = pd.read_csv(param.compatible_filename_sampling)
phi_vs_compatible_seq_metric_allstructures = load_df_into_dict_of_dict(df_comp, 'structure 1', 'structure 2', 'fraction comp seq', structure_list_seq_sample)

###############################################################################################
###############################################################################################
print(  'compatible set overlap - not relative to neutral set', flush=True)
###############################################################################################
###############################################################################################

def random_compatible_seq(structure, num_sample):
   sequence_sample_list = []
   bp_map = get_basepair_indices_from_dotbracket(structure)
   for i in range(num_sample):
      seq = ['0',] * len(structure)
      for siteindex, site in enumerate(structure):
         if site == '.':
            seq[siteindex] =  random.choice(['A', 'U', 'C', 'G'])
         elif site == '(':
            bp = random.choice(['AU', 'UA', 'CG', 'GC', 'GU', 'UG'])
            seq[siteindex] = bp[0]
            seq[bp_map[siteindex]] = bp[1]
      sequence_sample_list.append(''.join(seq))
   return sequence_sample_list

if not isfile(param.compatible_filename_sampling_not_neutral_set):
   phi_vs_compatible_seq_metric_allstructures2 = {p: {} for p in phi_vs_phi_pq_allstructures}
   for ph in phi_vs_compatible_seq_metric_allstructures2:
      print('ph', ph, flush=True)
      random_compatible_seq_list = random_compatible_seq(ph, num_sample=int(param.number_walks * param.length_per_walk/param.every_Nth_seq))
      print('start parallel', flush=True)
      pool_function = partial(fraction_compatible_seqlist, seq_list=deepcopy(random_compatible_seq_list))
      new_struct = [p for p in phi_vs_phi_pq_allstructures[ph].keys() if p!= '|']
      with Pool(25) as p:
         pool_result = p.map(pool_function, new_struct)
      for q, f in zip(new_struct, pool_result):
         phi_vs_compatible_seq_metric_allstructures2[ph][q] = f
   structure_one_list, structure_two_list, seq_comp_list = zip(*[(structureone, structuretwo, phi) for structureone in phi_vs_compatible_seq_metric_allstructures2 for structuretwo, phi in phi_vs_compatible_seq_metric_allstructures2[structureone].items()])
   df_comp2 = pd.DataFrame.from_dict({'structure 1': structure_one_list, 'structure 2': structure_two_list, 'fraction comp seq': seq_comp_list})
   df_comp2.to_csv(param.compatible_filename_sampling_not_neutral_set)
   
## load 
df_comp2 = pd.read_csv(param.compatible_filename_sampling_not_neutral_set)
phi_vs_compatible_seq_metric_allstructures2 = load_df_into_dict_of_dict(df_comp2, 'structure 1', 'structure 2', 'fraction comp seq', structure_list_seq_sample)


###############################################################################################
###############################################################################################
print( 'get correlations', flush=True)
###############################################################################################
###############################################################################################
for type_correlation, correlation_function in [('Pearson', logPearsonR_nonzero_values), ('top-30', partial(correct_top_x_predicted, x=30))]:
   ###############################################################################################
   print( 'get data: phi_pq vs mean Boltzmann probabilities  - correlations' )
   ###############################################################################################  
   corr_filename = './GPmapdata_fullstructures/Psampling_phenotype_phipq_correlations'+type_correlation+str(param.sample_filename)+'.csv'
   if not isfile(corr_filename):
      corr_Boltz_phipqS, corr_freq_phipqS = [], []
      for ph in structure_list_seq_sample:  
         structure_list_all = list(set([s for s in ph_vs_Boltzmann_freq_in_neutral_set_allstructures[ph] if s not in ['|', '.']] + [s for s in phi_vs_phi_pq_allstructures[ph] if s not in ['|', '.']] + [s for s in structure_vs_freq_estimate if s not in ['|', '.']]))
         boltz_freq_list = [ph_vs_Boltzmann_freq_in_neutral_set_allstructures[ph][s] if s in ph_vs_Boltzmann_freq_in_neutral_set_allstructures[ph] else 0 for s in structure_list_all if s != ph]
         freq_list = [structure_vs_freq_estimate[s] if s in structure_vs_freq_estimate else 0 for s in structure_list_all if s != ph]
         phipq_list_s = [phi_vs_phi_pq_allstructures[ph][s] if s in phi_vs_phi_pq_allstructures[ph] else 0 for s in structure_list_all if s != ph]
         corr_Boltz_phipqS.append(correlation_function(boltz_freq_list, phipq_list_s)[0])
         corr_freq_phipqS.append(correlation_function(freq_list, phipq_list_s)[0])
         del boltz_freq_list, freq_list, phipq_list_s
      df_corr =  pd.DataFrame.from_dict({'structure': structure_list_seq_sample, 
                                      type_correlation + ' correlation mean Boltzmann frequ. vs phiqp substitutions': corr_Boltz_phipqS,
                                      type_correlation + ' correlation structure frequ. vs phiqp substitutions': corr_freq_phipqS})
      df_corr.to_csv(corr_filename)
   else:
      df_corr = pd.read_csv(corr_filename)

###############################################################################################
###############################################################################################
print( 'get correlations', flush=True)
###############################################################################################
###############################################################################################
from KC import calc_KC
def dotbracket_to_bin(db_string):
  return ''.join([{'(': '10', ')': '01', '.': '00'}[c] for c in db_string])


for type_correlation, correlation_function in [('Pearson', logPearsonR_nonzero_values), ('top-30', partial(correct_top_x_predicted, x=30))]:
   ###############################################################################################
   print( 'get data: phi_pq vs mean Boltzmann probabilities  - correlations including similarity measures' )
   ###############################################################################################  
   corr_filename = './GPmapdata_fullstructures/Psampling_phenotype_phipq_correlations'+type_correlation+str(param.sample_filename)+'_with_sim.csv'
   if not isfile(corr_filename):
      corr_Boltz_phipqS, corr_freq_phipqS, corr_sim_phipqS, corr_seq_compatible_phipqS, corr_seq_compatible_phipqS_no_set, corr_AIT_phipqS = [], [], [], [], [], []
      for ph in structure_list_seq_sample:  
         structure_list_all = list(set([s for s in ph_vs_Boltzmann_freq_in_neutral_set_allstructures[ph] if s not in ['|', '.']] + [s for s in phi_vs_phi_pq_allstructures[ph] if s not in ['|', '.']] + [s for s in structure_vs_freq_estimate if s not in ['|', '.']]))
         #####
         print('get data', ph, flush=True)
         boltz_freq_list = [ph_vs_Boltzmann_freq_in_neutral_set_allstructures[ph][s] if s in ph_vs_Boltzmann_freq_in_neutral_set_allstructures[ph] else 0 for s in structure_list_all if s != ph]
         freq_list = [structure_vs_freq_estimate[s] if s in structure_vs_freq_estimate else 0 for s in structure_list_all if s != ph]
         bp_dist_list_for_plot_RNA = [2 * len(s) - RNA.bp_distance(s, ph) for s in structure_list_all if s != ph]
         phipq_list_s = [phi_vs_phi_pq_allstructures[ph][s] if s in phi_vs_phi_pq_allstructures[ph] else 0 for s in structure_list_all if s != ph]
         comp_seq_list_for_plot_RNA = [phi_vs_compatible_seq_metric_allstructures[ph][s] if s in phi_vs_compatible_seq_metric_allstructures[ph] else 0 for s in structure_list_all if s != ph]
         comp_seq_list_for_plot_RNA2 = [phi_vs_compatible_seq_metric_allstructures2[ph][s] if s in phi_vs_compatible_seq_metric_allstructures2[ph] else 0 for s in structure_list_all if s != ph]
         Dingle_pred_list = [100 - (calc_KC(dotbracket_to_bin(s+ph)) - calc_KC(dotbracket_to_bin(ph))) for s in structure_list_all if s != ph]
         corr_Boltz_phipqS.append(correlation_function(boltz_freq_list, phipq_list_s)[0])
         corr_freq_phipqS.append(correlation_function(freq_list, phipq_list_s)[0])
         corr_seq_compatible_phipqS.append(correlation_function(comp_seq_list_for_plot_RNA, phipq_list_s)[0])
         corr_seq_compatible_phipqS_no_set.append(correlation_function(comp_seq_list_for_plot_RNA2, phipq_list_s)[0])
         if type_correlation == 'Pearson':
            corr_sim_phipqS.append(linlogPearsonR_nonzero_values(bp_dist_list_for_plot_RNA, phipq_list_s)[0])
            corr_AIT_phipqS.append(linlogPearsonR_nonzero_values(Dingle_pred_list, phipq_list_s)[0])
         else:
            corr_sim_phipqS.append(correlation_function(bp_dist_list_for_plot_RNA, phipq_list_s)[0])
            corr_AIT_phipqS.append(correlation_function(Dingle_pred_list, phipq_list_s)[0])            
         del boltz_freq_list, freq_list, phipq_list_s
      df_corr =  pd.DataFrame.from_dict({'structure': structure_list_seq_sample, 
                                      type_correlation + ' correlation mean Boltzmann frequ. vs phiqp substitutions': corr_Boltz_phipqS,
                                      type_correlation + ' correlation structure frequ. vs phiqp substitutions': corr_freq_phipqS,
                                      type_correlation + ' correlation fraction sequence compatibility vs phiqp substitutions': corr_seq_compatible_phipqS,
                                      type_correlation + ' correlation fraction sequence compatibility vs phiqp substitutions (not neutral set)': corr_seq_compatible_phipqS_no_set,
                                      type_correlation + ' correlation (1 - conditional complexity) vs phiqp substitutions': corr_AIT_phipqS,
                                      type_correlation + ' correlation (1 - bp.dist) vs phiqp substitutions': corr_sim_phipqS})
      df_corr.to_csv(corr_filename)
   else:
      df_corr = pd.read_csv(corr_filename)
###############################################################################################
###############################################################################################
print(  'get Hamming distance distribution', flush=True)
###############################################################################################
###############################################################################################
from collections import Counter

def get_Hamming_dist_distribution(seq_list):
   hamming_dist_vs_count = {}
   for seqindex, seq1 in enumerate(seq_list[:10**4]):#[:10**4]
      for seq2 in seq_list[:seqindex]:
         h = Hamming_dist(seq1, seq2)
         try:
            hamming_dist_vs_count[h] += 1
         except KeyError:
            hamming_dist_vs_count[h] = 1
   return hamming_dist_vs_count
if not isfile(param.Hammingdist_neutral_set_filename):
   structure_sample = sample.read_structure_list(param.filename_structure_list)

   ###
   filename_seq_sample = './GPmapdata_fullstructures/sequence_sample'+'Boltzmann' +param.sample_filename+'mfe_unique.csv'
   structure_sample_df = pd.read_csv(filename_seq_sample)
   struct_vs_seq = {structure: [] for structure in structure_sample}
   for i, row in structure_sample_df.iterrows(): 
      struct_vs_seq[row['structure']].append(row['seq'])
   print('loaded sequence sample')
   with Pool(15) as p:
      pool_result = p.map(get_Hamming_dist_distribution, [deepcopy(struct_vs_seq[s]) for s in structure_sample])
   structure_vs_dist_list = {s: Hamming_result for s, Hamming_result in zip(structure_sample, pool_result)}
   structure_list_hamming, hamming_list, hfreq_list = zip(*[(structureone, dist, f) for structureone in structure_sample for dist, f in structure_vs_dist_list[structureone].items()])
   df_Hamming = pd.DataFrame.from_dict({'structure': structure_list_hamming, 'Hamming dist': hamming_list, 'frequency': hfreq_list})
   df_Hamming.to_csv(param.Hammingdist_neutral_set_filename)

