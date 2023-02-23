#!/usr/bin/env python3
import numpy as np

L_sampling, K = 30, 4 
ncpus = 25
number_walks, length_per_walk, every_Nth_seq = 100, 10**5, 50  #100, 10**4, 20 
max_no_trials_RNAinverse = 10
########
num_sample_str = 50
#num_sample_str_plot2 = 10**3
number_measurements = 3
sample_size_gsample = 10**8 #10**8
num_sample = int(num_sample_str)
######
filename_structure_sample = str(L_sampling) + '_Nsamples' + str(num_sample) +'gsample' + str(int(np.log10(sample_size_gsample)))
sample_filename = filename_structure_sample +'_'+str(number_walks)+'_'+str(length_per_walk)+'_'+str(every_Nth_seq )+'_'+str(max_no_trials_RNAinverse)
filename_structure_list = './GPmapdata_fullstructures/structure_sample'+filename_structure_sample+'.txt'
######
phipq_filename_sampling = './GPmapdata_fullstructures/phipq_'+sample_filename+'mfe_unique.csv'
meanBoltzmann_neutral_set_filename_sampling = './GPmapdata_fullstructures/meanBoltzmann_neutral_set_'+sample_filename+'mfe_unique.csv'
Hammingdist_neutral_set_filename = './GPmapdata_fullstructures/Hamming_dist_in_neutral_set_neutral_set_'+sample_filename+'mfe_unique.csv'
compatible_filename_sampling = './GPmapdata_fullstructures/compatible_seq_neutral_set_'+sample_filename+'mfe_unique.csv'
compatible_filename_sampling_not_neutral_set = './GPmapdata_fullstructures/compatible_seq_'+sample_filename+'mfe_unique.csv'
###
neutral_set_size_filename_sampling = './GPmapdata_fullstructures/neutral_set_size_'+filename_structure_sample+'gsample' + str(int(np.log10(sample_size_gsample)))+'mfe_unique.csv'
meanBoltzmann_neutral_set_filename_gsampling = './GPmapdata_fullstructures/gsample_test_meanBoltzmann_neutral_set_'+filename_structure_sample+'gsample' + str(int(np.log10(sample_size_gsample * 10)))+'mfe_unique.csv'
phipq_filename_gsampling = './GPmapdata_fullstructures/gsample_test_phipq_'+filename_structure_sample+'gsample' + str(int(np.log10(sample_size_gsample * 10)))+'mfe_unique.csv'
###
Nresult_filename = './GPmapdata_fullstructures/neutral_set_size_prediction' + filename_structure_sample +'nm'+str(number_measurements)+ 'gsample'+str(int(np.log10(sample_size_gsample//10))) +   '.csv'
