import numpy as np
from .thermodynamic_functions import get_unique_mfe_structure_seq_str
import subprocess
import RNA
from .general_functions import neighbours_g_given_site, bp_swaps_g_given_site, neighbours_g
import random
import networkx as nx
from .rna_structural_functions import  is_likely_to_be_valid_structure, dotbracket_to_coarsegrained
from multiprocessing import Pool
from os.path import isfile
from copy import deepcopy
import random



############################################################################################################
## get sample of structures
############################################################################################################



def save_structure_list(structure_sample, filename):
   """save list of structures to file, one structure per line"""
   with open(filename, 'w') as textfile:
      for structure in structure_sample:
         textfile.write(structure+'\n')

def read_structure_list(filename, allow_isolated_bps=False):
   """read list of structures from file"""
   with open(filename, 'r') as textfile:
      structure_sample = [str(line.strip()) for line in textfile.readlines()]
   for structure in structure_sample:
      validity_structure = is_likely_to_be_valid_structure(structure, allow_isolated_bps=allow_isolated_bps)
      if not validity_structure:
         print( structure, 'does not seem to be a valid structure')
      assert validity_structure
   return structure_sample


############################################################################################################
## run RNAinverse
############################################################################################################

def find_start_sequence(dotbracket_structure, max_no_trials=5):
   """run inverse_fold until a sequence whose minimum free energy is the given structure,
   with an energy gap to the next lowest-lying structure of > 0 (doe to ViennaRNA energy rounding to first decimal)
   if no result is found, run at most max_no_trials times"""
   L = len(dotbracket_structure)
   for trial in range(max_no_trials):
      random_startseq_int = ''.join(random.choices(['A', 'G', 'C', 'U'], k=L))
      (sequence_str, distance_from_target_str) = RNA.inverse_fold(random_startseq_int, dotbracket_structure)
      if distance_from_target_str < 0.01 and get_unique_mfe_structure_seq_str(sequence_str) == dotbracket_structure:
         return sequence_str
      elif distance_from_target_str < 0.01 and not get_unique_mfe_structure_seq_str(sequence_str) == '|':
         print('problem with inverse folding', sequence_str, dotbracket_structure, distance_from_target_str, get_unique_mfe_structure_seq_str(sequence_str))
   return ''



############################################################################################################
## site scanning method
############################################################################################################
## site scanning method adapted from the description Weiss, Marcel; Ahnert, Sebastian E. (2020): 
## Supplementary Information from Using small samples to estimate neutral component size and robustness in the genotype-phenotype map 
## of RNA secondary structure. The Royal Society. Journal contribution. https://doi.org/10.6084/m9.figshare.12200357.v2
## here base pair swaps are used as well as point mutations

def rw_sitescanning(startseq, length_per_walk, every_Nth_seq):
   """perform a site-scanning random walk of length length_per_walk starting from startseq (tuple of ints) and subsample every_Nth_seq;
   site-scanning method following Weiss and Ahnert (2020), Royal Society Interface
   neutrality is defined based on RNAfold alone (without checking if mfe level degenerate)"""
   seq_list_rw, current_seq, dotbracket_structure = [], deepcopy(startseq), get_unique_mfe_structure_seq_str(startseq)
   L, site_to_scan = len(startseq), 0
   seq_list_rw.append(current_seq)
   neutral_neighbours_startseq = [g_mut for g_mut in neighbours_g(current_seq, L) if get_unique_mfe_structure_seq_str(startseq) == dotbracket_structure]
   if len(neutral_neighbours_startseq) == 0:
      return [current_seq,]
   while len(seq_list_rw) < length_per_walk:
      if dotbracket_structure[site_to_scan] == '.':
         neighbours_given_site = neighbours_g_given_site(current_seq, L, site_to_scan)
      else:
         neighbours_given_site = bp_swaps_g_given_site(current_seq, site_to_scan, dotbracket_structure)
      neighbours_given_site_shuffled = [deepcopy(neighbours_given_site[i]) for i in random.sample(range(len(neighbours_given_site)), k=len(neighbours_given_site))]
      for g in neighbours_given_site_shuffled:
         if dotbracket_structure == get_unique_mfe_structure_seq_str(g):
            current_seq = deepcopy(g)
            seq_list_rw.append(current_seq)
            break
      site_to_scan = (site_to_scan+1)%L  
   assert len(seq_list_rw) == length_per_walk
   return [deepcopy(seq_list_rw[i]) for i in random.sample(range(length_per_walk), k=length_per_walk//every_Nth_seq)]
      
def get_x_random_walks(dotbracket_structure, length_per_walk, every_Nth_seq, number_walks):
   """execute rw_sitescanning for dotbracket_structure number_walks times;
   length_per_walk and every_Nth_seq are passed directly to the site-scanning random walk function"""
   print('start site-scanning for', dotbracket_structure, flush=True)
   start_sequences, seq_list_rw = [], []
   while len(start_sequences) < number_walks:
      RNAinverse_result = find_start_sequence(dotbracket_structure)
      if len(RNAinverse_result): #empty tuple if RNAinverse fails
         start_sequences.append(RNAinverse_result)
         assert get_unique_mfe_structure_seq_str(RNAinverse_result) == dotbracket_structure
   for startseq in start_sequences:
      seq_list_rw += rw_sitescanning(startseq, length_per_walk, every_Nth_seq) 
   return seq_list_rw


