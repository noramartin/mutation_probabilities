import numpy as np
from functools import partial
from .thermodynamic_functions import get_unique_mfe_structure_seq_str, get_dotbracket_from_int, get_unique_mfe_structure_as_int_seq_str, get_Boltzmann_freq_list, get_all_mfe_structures_seq_str
from .rna_structural_functions import sequence_compatible_with_basepairs, dotbracket_to_coarsegrained
from multiprocessing import Pool
import RNA
import random
from copy import deepcopy
from collections import Counter

def g_sampling_struct_sample_balance_n_stacks(L, final_sample_size, gsample_size, max_sample_at_once= 10**4):
   sample_finished = 0
   structures = set([])
   sampling_function = partial(get_structure_random_seq, mfe_function=get_unique_mfe_structure_as_int_seq_str, L=L)
   while sample_finished < gsample_size:
      next_sample_size = min(max_sample_at_once, gsample_size - sample_finished)
      print( 'g_sampling_Nestimate_mfe: finished {:.2e}'.format(sample_finished), 'sequences, next sample size: {:.2e}'.format(next_sample_size), ', total: {:.2e}'.format(gsample_size))
      sample_finished += next_sample_size
      with Pool(25) as p:
         pool_result = p.map(sampling_function, np.arange(next_sample_size))
      assert len(pool_result) == next_sample_size
      for structure in pool_result:
         structures.add(structure)
      del pool_result
   structure_list_dotbracket = [get_dotbracket_from_int(structure) for structure in structures if get_dotbracket_from_int(structure).count('(') > 0]
   nstacks_list = [dotbracket_to_coarsegrained(s).count('[') for s in structure_list_dotbracket]
   nstacks_list_counts = Counter(nstacks_list)
   weight_in_sample = [1/nstacks_list_counts[n] for n in nstacks_list]
   return np.random.choice(structure_list_dotbracket, final_sample_size, replace=False, p=np.divide(weight_in_sample, sum(weight_in_sample)))


def g_sampling_seq_sample_mfe(structure_list, sample_size, max_sample_at_once= 10**4, seq_per_structure = 10**3):
   sample_finished, L = 0, len(structure_list[0])
   structure_vs_seq = {structure: [] for structure in structure_list}
   mfe_function = partial(get_unique_mfe_structure_seq_str)
   while sample_finished < sample_size:
      next_sample_size = min(max_sample_at_once, sample_size - sample_finished)
      print( 'g_sampling_Nestimate_mfe: finished {:.2e}'.format(sample_finished), 'sequences, next sample size: {:.2e}'.format(next_sample_size), ', total: {:.2e}'.format(sample_size))
      sample_finished += next_sample_size
      sequence_sample_list = [''.join(np.random.choice(['A', 'C', 'G', 'U'], replace=True, size=L)) for i in range(next_sample_size)]     
      with Pool(25) as p:
         pool_result = p.map(mfe_function, sequence_sample_list)
      assert len(pool_result) == next_sample_size
      for structure in structure_list:
         if len(structure_vs_seq[structure]) < seq_per_structure and pool_result.count(structure) > 0:
            for seq, struct in zip(sequence_sample_list, pool_result):
               if struct == structure:
                  structure_vs_seq[structure].append(deepcopy(seq))
      del pool_result, sequence_sample_list
   return structure_vs_seq

def get_structure_random_seq(i, mfe_function, L):
   seq = ''.join(random.choices(['A', 'C', 'G', 'U'], k=L))
   return mfe_function(seq)

def g_sampling_Nestimate_mfe_allstructures(L, sample_size, max_sample_at_once= 10**4, min_count = 5):
   sample_finished = 0
   structure_vs_N = {}
   sampling_function = partial(get_structure_random_seq, mfe_function=get_unique_mfe_structure_as_int_seq_str, L=L)
   while sample_finished < sample_size:
      next_sample_size = min(max_sample_at_once, sample_size - sample_finished)
      print( 'g_sampling_Nestimate_mfe: finished {:.2e}'.format(sample_finished), 'sequences, next sample size: {:.2e}'.format(next_sample_size), ', total: {:.2e}'.format(sample_size))
      sample_finished += next_sample_size
      with Pool(25) as p:
         pool_result = p.map(sampling_function, np.arange(next_sample_size))
      assert len(pool_result) == next_sample_size
      for structure in pool_result:
         try:
            structure_vs_N[structure] += 1
         except KeyError:
            structure_vs_N[structure] = 1
      del pool_result
   return {get_dotbracket_from_int(structure): count/float(sample_size) for structure, count in structure_vs_N.items() if count >= min_count}

def g_sampling_Boltzmannprob_mfe(structure_list, sample_size, max_sample_at_once= 10**4, ncpus=5):
   sample_finished, L = 0, len(structure_list[0])
   structure_vs_p = {structure: [] for structure in structure_list}
   p_function = partial(get_Boltzmann_freq_list, structure_dotbracket_list=structure_list)   
   assert sample_size % max_sample_at_once == 0 #need samples of even size to be able to average  
   while sample_finished < sample_size:
      next_sample_size = min(max_sample_at_once, sample_size - sample_finished)
      sample_finished += next_sample_size
      print( 'g_sampling_Nestimate_mfe: finished {:.2e}'.format(sample_finished), 'sequences, next sample size: {:.2e}'.format(next_sample_size), ', total: {:.2e}'.format(sample_size))
      sampling_function = partial(get_structure_random_seq, mfe_function=p_function, L=L)
      with Pool(processes = ncpus) as p:
         pool_result = p.map(sampling_function, np.arange(next_sample_size))
      assert len(pool_result) == next_sample_size
      for structureindex, structure in enumerate(structure_list):
         structure_vs_p[structure].append(np.mean([res[structureindex] for res in pool_result]))
      del pool_result 
      print( 'finished ' + str(sample_finished) + ' out of ' + str(sample_size) + ' for g-sample')
   return {structure: np.mean(p) for structure, p in structure_vs_p.items()}

def g_sampling_Boltzmannprob_no_mfe(structure_list, sample_size, max_sample_at_once= 10**4, ncpus=5):
   sample_finished, L = 0, len(structure_list[0])
   structure_vs_p = {structure: [] for structure in structure_list}
   structure_vs_nseq = {structure: [] for structure in structure_list}
   p_function = partial(get_Boltzmann_freq_list, structure_dotbracket_list=structure_list)     
   assert sample_size % max_sample_at_once == 0 #need samples of even size to be able to average  
   while sample_finished < sample_size:
      next_sample_size = min(max_sample_at_once, sample_size - sample_finished)
      sample_finished += next_sample_size
      print( 'g_sampling_Nestimate_mfe: finished {:.2e}'.format(sample_finished), 'sequences, next sample size: {:.2e}'.format(next_sample_size), ', total: {:.2e}'.format(sample_size))
      sequence_sample_list = [''.join(np.random.choice(['A', 'C', 'G', 'U'], replace=True, size=L)) for i in range(next_sample_size)]
      with Pool(processes = ncpus) as p:
         prob_structures_result = p.map(p_function, sequence_sample_list)
      with  Pool(processes = ncpus) as p:
         mfe_result = p.map(get_all_mfe_structures_seq_str, sequence_sample_list)
      assert len(prob_structures_result) == next_sample_size
      for structureindex, structure in enumerate(structure_list):
         structure_vs_p[structure].append(np.mean([res[structureindex]  for seqindex, res in enumerate(prob_structures_result) if  structure not in mfe_result[seqindex]]))
         structure_vs_nseq[structure].append(len([seqindex  for seqindex, res in enumerate(prob_structures_result) if  structure not in mfe_result[seqindex]]))
      del prob_structures_result , sequence_sample_list, mfe_result
      print( 'finished ' + str(sample_finished) + ' out of ' + str(sample_size) + ' for g-sample')
   return {structure: sum([p*count for p, count in zip(p_list, structure_vs_nseq[structure])])/sum(structure_vs_nseq[structure]) for structure, p_list in structure_vs_p.items()}

