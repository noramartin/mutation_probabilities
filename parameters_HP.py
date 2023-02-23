#!/usr/bin/env python3
import numpy as np
from copy import deepcopy
from HP_model.HPfunctions import *
from os.path import isfile
import pandas as pd


n = 5
K = 2
kbT = 0.5 #0.1 # 1
number_sequences_per_structure = 10**3
g_sample_size_meanBoltz_random_seq = 10**5
g_sample_size_hist = 10**3
L = n * n
###################################################################################################
GPmap_filename = './GPmapdata_HP/GPmapHP_L'+str(L)+'.npy'
neutral_set_size_filename = './GPmapdata_HP/HPneutral_set_size_L'+str(L)+'.csv'
phipq_filename = './GPmapdata_HP/HPphipq_L'+str(L)+'.csv'
meanBoltzmann_neutral_set_filename = './GPmapdata_HP/HPmeanBoltzmann_neutral_set_L'+str(L)+'_10kbT'+str(int(kbT * 10)) +'_'+str(number_sequences_per_structure)+'.csv'
Hammingdist_neutral_set_filename = './GPmapdata_HP/Hamming_dist_neutral_set_L'+str(L)+'_10kbT'+str(int(kbT * 10)) +'_'+str(number_sequences_per_structure)+'.csv'
contact_map_list_filename = './GPmapdata_HP/contact_map_listHP_L'+str(L)+'.csv'
###################################################################################################
###################################################################################################
###################################################################################################
#usually structures are referred to as the index in this list + 1 (so zero is undefined)
if not isfile(contact_map_list_filename):
    contact_map_vs_updown = {up_down_to_contact_map(structure_up_down_notation): deepcopy(structure_up_down_notation) for structure_up_down_notation in all_structures_with_unique_contact_maps(n)}
    contact_map_list = [deepcopy(c) for c in sorted(contact_map_vs_updown.keys())]
    df_CM = pd.DataFrame.from_dict({'lower contacts of structure': ['_'+'_'.join([str(c[0]) for c in CM]) for CM in contact_map_list], 
   	                               'corresponding upper contacts of structure': ['_'+'_'.join([str(c[1]) for c in CM]) for CM in contact_map_list],
   	                               'up-down string of structure': ['_'+'_'.join([str(c) for c in contact_map_vs_updown[CM]]) for CM in contact_map_list]})
    df_CM.to_csv(contact_map_list_filename)
else:
	df_CM = pd.read_csv(contact_map_list_filename)
	contact_map_list = [tuple([(int(c0), int(c1)) for c0, c1 in zip(row['lower contacts of structure'].strip('_').split('_'), row['corresponding upper contacts of structure'].strip('_').split('_'))]) if len(row['lower contacts of structure']) > 1 else tuple([]) for rowi, row in df_CM.iterrows()]
	contact_map_vs_updown = {deepcopy(contact_map_list[rowi]): [int(ud) for ud in row['up-down string of structure'].strip('_').split('_')] for rowi, row in df_CM.iterrows()}
print('contact maps',  len(contact_map_vs_updown))

