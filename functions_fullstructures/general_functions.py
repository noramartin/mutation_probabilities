import numpy as np
from scipy.stats import pearsonr, spearmanr
from copy import deepcopy
from .rna_structural_functions import get_basepair_indices_from_dotbracket

def Hamming_dist(seq1, seq2):
   assert len(seq1) == len(seq2)
   return len([i for i in range(len(seq1)) if seq1[i] != seq2[i]])

def load_df_into_dict_of_dict(df, column1, column2, column3, keys):
   dict_output = {k: {} for k in keys}
   for rowindex, row in df.iterrows():
      dict_output[row[column1]][row[column2]] = row[column3]
   return deepcopy(dict_output)

def correct_top_x_predicted(predicted_values, true_values, x, return_indices=False):
   if len([l for l in predicted_values if l > 0]) < 2 * x or len([l for l in true_values if l > 0]) < 2 * x:
      if return_indices:
         return np.nan, np.nan, [], []
      return np.nan, np.nan
   indices_true = sorted(list(range(len(true_values))), key={i: v for i, v in enumerate(true_values)}.get, reverse=True)[:x]
   indices_predicted = sorted(list(range(len(predicted_values))), key={i: v for i, v in enumerate(predicted_values)}.get, reverse=True)[:x]
   assert len(indices_true) == len(indices_predicted) == x
   assert np.argmax(predicted_values) in indices_predicted and np.argmax(true_values) in indices_true
   if abs(sorted(predicted_values, reverse=True)[x-1]/sorted(predicted_values, reverse=True)[x] - 1) < 0.001:
      print('there is a tie', sorted(predicted_values, reverse=True)[x-1], sorted(predicted_values, reverse=True)[x])
   if abs(sorted(true_values, reverse=True)[x-1]/sorted(true_values, reverse=True)[x] - 1) < 0.001:
      print('there is a tie', sorted(true_values, reverse=True)[x-1], sorted(true_values, reverse=True)[x])
   result = len([i for i in indices_predicted if i in indices_true])/x
   if return_indices:
      return result, result, indices_true, indices_predicted
   return result, result

def SpearmanR_nonzero_values(x_list, y_list):
  return spearmanr([x for i, x in enumerate(x_list) if x > 0 and y_list[i] > 0], [y for i, y in enumerate(y_list) if y > 0 and x_list[i] > 0])


def logPearsonR_nonzero_values(x_list, y_list):
  return pearsonr(np.log([x for i, x in enumerate(x_list) if x > 0 and y_list[i] > 0]), np.log([y for i, y in enumerate(y_list) if y > 0 and x_list[i] > 0]))

def linlogPearsonR_nonzero_values(x_list, y_list):
  return pearsonr([x for i, x in enumerate(x_list) if y_list[i] > 0], np.log([y for i, y in enumerate(y_list) if y > 0]))


def get_phipq_sequence_sample(sequence_list, folding_function, min_phi=0):
   print('start phipq calculation', flush=True)
   phi_pq = {}
   L = len(sequence_list[0])
   for seqindex, seq in enumerate(sequence_list):
      neighbours = neighbours_g(seq, L)
      for neighbourgeno in neighbours:
         neighbourpheno = folding_function(neighbourgeno)
         try:
            phi_pq[neighbourpheno] += 1.0/(len(sequence_list) * 3 * L)
         except KeyError:
            phi_pq[neighbourpheno] = 1.0/(len(sequence_list) * 3 * L)  
   return {ph: phi for ph, phi in phi_pq.items() if phi >= min_phi}

def get_boltzmann_sequence_sample(sequence_list, boltzmann_ensemble_function, min_Boltz=0):
   print('start boltzmann calculation', flush=True)
   boltz_q = {}
   L = len(sequence_list[0])
   for seqindex, seq in enumerate(sequence_list):
      for structure, f in boltzmann_ensemble_function(seq).items():    
         try:
            boltz_q[deepcopy(structure)] += f/float(len(sequence_list))
         except KeyError:
            boltz_q[deepcopy(structure)] = f/float(len(sequence_list))  
   return {ph: b for ph, b in boltz_q.items() if b >= min_Boltz}

############################################################################################################
## mutational neighbourhood
############################################################################################################
allowed_basepairs = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG']

def neighbours_g(g, L): 
   """list all pont mutational neighbours of sequence g (integer notation)"""
   return [''.join([oldK if gpos!=pos else new_K for gpos, oldK in enumerate(g)]) for pos in range(L) for new_K in ['A', 'C', 'G', 'U'] if g[pos]!=new_K]

def neighbours_g_given_site(g, L, pos): 
   """list all pont mutational neighbours of sequence g (integer notation) with a substitution at position pos"""
   return [''.join([oldK if gpos!=pos else new_K for gpos, oldK in enumerate(g)]) for new_K in ['A', 'C', 'G', 'U'] if g[pos]!=new_K]

def bp_swaps_g_given_site(g, pos, structure_dotbracket): 
   """perform all base pair swaps at the given position and its paired position"""
   g_list = []
   assert structure_dotbracket[pos] in ['(', ')']
   paired_pos = get_basepair_indices_from_dotbracket(structure_dotbracket)[pos]
   for new_bp in allowed_basepairs:
      if new_bp[0] != g[pos] or new_bp[1] != g[paired_pos]: # not identical bp 
         g_new = [x for x in g]
         g_new[pos] = new_bp[0]
         g_new[paired_pos] = new_bp[1]
         g_list.append(''.join(g_new))
   return g_list

############################################################################################################
## test
############################################################################################################
if __name__ == "__main__":
   print( 'test: general_functions.py')
   l1 = [8, 3, 1, 9, 3, 4, 2, 5, 2]
   l2 = [1, 2, 3, 9, 5, 6, 7, 8, 9]
   assert abs(2/3 - correct_top_x_predicted(l2, l1, 3)[0]) < 0.001
   assert abs(1/2 - correct_top_x_predicted(l2, l1, 4)[0]) < 0.001
   print( '\n\n-------------\n\n')
   assert len(neighbours_g_given_site('ACGUCGG', 7, 2)) == 3
   assert len(neighbours_g('ACGUCGG', 7)) == 21
   for n in neighbours_g('ACGUCGG', 7) + neighbours_g_given_site('ACGUCGG', 7, 2):
      assert n != 'ACGUCGG'
   assert len(bp_swaps_g_given_site('ACGUCGG', 1, '.(...).')) == 5
   for n in bp_swaps_g_given_site('ACGUCGG', 1, '.(...).'):
      assert n != 'ACGUCGG' and (n[1] != 'C' or n[-2] != 'G') and 1 <= len([x for i, x in enumerate(n) if x != 'ACGUCGG'[i]]) <= 2 and n[1] + n[-2] in allowed_basepairs


   print( '\n\n-------------\n\n')



