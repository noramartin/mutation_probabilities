#!/usr/bin/env python3
import numpy as np
import pandas as pd
from os.path import isfile
import parameters_HP as param
from HP_model.HPfunctions import *



def get_structure_vs_mean_Boltz(nsample, kbT):
   str_vs_Boltz = {s: [] for s in range(1, len(param.contact_map_list) + 1)}
   for i in range(nsample):
      seq = [i for i in np.random.choice([0, 1], param.L, replace=True)]
      Boltz_list = HPget_Boltzmann_freq_list(seq, param.contact_map_list, kbT=kbT)
      for i, s in enumerate(param.contact_map_list):
         str_vs_Boltz[i + 1].append(Boltz_list[i])
   return {s: np.mean(l) for s, l in str_vs_Boltz.items()}


def get_structure_vs_mean_Boltz_no_mfe(nsample, kbT):
   str_vs_Boltz = {s: [] for s in range(1, len(param.contact_map_list) + 1)}
   n_samples_finished = 0
   while n_samples_finished < nsample:
      seq = [i for i in np.random.choice([0, 1], param.L, replace=True)]
      #mfe_struct = find_mfe(seq, param.contact_map_list)
      #if mfe_struct > 0.5:
      Boltz_list = HPget_Boltzmann_freq_list(seq, param.contact_map_list, kbT=kbT)
      free_energy_list = [free_energy(seq, contact_map) for contact_map in param.contact_map_list]
      mfe_value = min(free_energy_list)
      for i, s in enumerate(param.contact_map_list):
         if abs(mfe_value - free_energy_list[i]) > 0.001:
            str_vs_Boltz[i + 1].append(Boltz_list[i])
      n_samples_finished += 1
   return {s: np.mean(l) for s, l in str_vs_Boltz.items()}

for type_Boltzmann_average in ['', 'no_mfe']:
   for kbT in [1, 0.5, 0.1]:
      ###############################################################################################
      ###############################################################################################
      meanBoltzmann_all_filename = './GPmapdata_HP/HPmeanBoltzmann_gsamplet_L'+str(param.L)+type_Boltzmann_average+'_100kbT'+str(int(100*kbT))  +'_'+str(param.g_sample_size_meanBoltz_random_seq)+'.csv'
      ###############################################################################################
      ###############################################################################################
      print( 'estimate mean Boltzmann frequencies by sampling', flush=True)
      ###############################################################################################
      ###############################################################################################
      if not isfile(meanBoltzmann_all_filename):
         if type_Boltzmann_average == '':
            structure_vs_mean_Boltz = get_structure_vs_mean_Boltz(param.g_sample_size_meanBoltz_random_seq, kbT=kbT)
         elif type_Boltzmann_average == 'no_mfe':
            structure_vs_mean_Boltz = get_structure_vs_mean_Boltz_no_mfe(param.g_sample_size_meanBoltz_random_seq, kbT=kbT)
         df_N = pd.DataFrame.from_dict({'structure': list(structure_vs_mean_Boltz.keys()), 'mean Boltzmann freq': [structure_vs_mean_Boltz[s] for s in structure_vs_mean_Boltz.keys()]})
         df_N.to_csv(meanBoltzmann_all_filename)
###############################################################################################
###############################################################################################
print( 'what is the g-sampled Boltzmann distribution of mfe structures at different temperatures', flush=True)
###############################################################################################
###############################################################################################

def get_mfe_Boltz_dist_gsample(nsample, kbT):
   boltz_list = []
   n_samples_finished = 0
   while n_samples_finished < nsample:
      seq = [i for i in np.random.choice([0, 1], param.L, replace=True)]
      mfe_struct = find_mfe(seq, param.contact_map_list)
      if mfe_struct > 0.5:
         Boltz_list = HPget_Boltzmann_freq_list(seq, param.contact_map_list, kbT=kbT)
         boltz_list.append(Boltz_list[mfe_struct - 1])
         n_samples_finished += 1
   return boltz_list

for kbT in [1, 0.5, 0.1]:
   print('kbT=' + str(kbT))
   gsampled_dist_filename = './GPmapdata_HP/HP_gsample_Boltzmann_dist_L'+str(param.L)+'_100kbT'+str(int(100*kbT))  +'_'+str(param.g_sample_size_hist)+'.npy'
   boltz_list = get_mfe_Boltz_dist_gsample(param.g_sample_size_hist, kbT)
   np.save(gsampled_dist_filename, boltz_list)
   