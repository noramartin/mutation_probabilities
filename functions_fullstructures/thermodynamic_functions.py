import RNA
import numpy as np
from .rna_structural_functions import sequence_compatible_with_basepairs, has_length_one_stack
from os.path import isfile
from copy import deepcopy

db_to_bin = {'.': '00', '(': '10', ')': '01', '_': '00', '[': '10', ']': '01'}

RNA.cvar.uniq_ML = 1 # global switch for unique multiloop decomposition
model = RNA.md()
model.noLP = 1 # no isolate base pairs
model.pf_smooth = 0 # deactivate partition function smoothing
kbT_RNA = RNA.exp_param().kT/1000.0 ## from https://github.com/ViennaRNA/ViennaRNA/issues/58
###############################################################################################
## Boltzmann frequency predictions
###############################################################################################


def get_Boltzmann_freq_list(seq, structure_dotbracket_list):
   ensemble = get_Boltzmann_freq_dict_lowenergyrange(seq)
   p_list = []
   for s in structure_dotbracket_list:
      try:
         p_list.append(ensemble[s])
      except KeyError:
         p_list.append(0)
   return p_list

def get_Boltzmann_freq_list_all_lowenergy_structures(seq, structure_dotbracket_list):
   """calculate Boltzman distribution - compute partition function manually, so isolared bps disabled"""
   a = RNA.fold_compound(seq, model)
   (mfe_structure, mfe) = a.mfe()
   a.exp_params_rescale(mfe)
   (prob_vector, dG) = a.pf()
   energy_list = [a.eval_structure(s) for s in structure_dotbracket_list]
   weight_list = np.exp(np.array(energy_list) * -1.0/kbT_RNA)
   Z = np.sum(weight_list)
   return [w/Z for w in weight_list]

def get_Boltzmann_freq_dict_lowenergyrange(seq, G_range_kcal = 15, test_no_isolated_bp= True):
   """calculate Boltzman-based stability"""
   structure_list = [deepcopy(s) for s in return_structures_in_energy_range(G_range_kcal, seq)] 
   for s in structure_list:
      assert not has_length_one_stack(s) or not test_no_isolated_bp
   return {s:p for s, p in zip(structure_list, get_Boltzmann_freq_list_all_lowenergy_structures(seq, structure_list))}
  
def return_structures_in_energy_range(G_range_kcal, seq):
   """ return structures in energy range (kcal/mol) of G_range_kcal from the mfe of sequence sequence_indices;
   code adapted from ViennaRNA documentation: https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/RNAlib-2.4.14.pdf;
   subopt is run for a small energy range at first and then a larger one, until a suboptimal structure is identified"""
   RNA.cvar.uniq_ML, structure_vs_energy = 1, {} # Set global switch for unique multiloop decomposition
   fold_compound_seq = RNA.fold_compound(seq, model)
   (mfe_structure, mfe) = fold_compound_seq.mfe()
   fold_compound_seq.subopt_cb(int(G_range_kcal*100.0*1.1), convert_result_to_dict, structure_vs_energy)
   subopt_structure_vs_G = {alternativestructure: fold_compound_seq.eval_structure(alternativestructure) for alternativestructure in structure_vs_energy 
                               if abs(fold_compound_seq.eval_structure(alternativestructure)-mfe) <= G_range_kcal}
   subopt_structure_vs_G[mfe_structure] = mfe
   return subopt_structure_vs_G

def convert_result_to_dict(structure, energy, data):
   """ function needed for subopt from ViennaRNA documentation: https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/RNAlib-2.4.14.pdf"""
   if not structure == None:
      data[structure] = energy
###############################################################################################
## focus on minimum-free energy structure and energy: 
## not tested whether there is an energy gap between top two structures
###############################################################################################

def get_unique_mfe_structure_seq_str(seq):
   """get minimum free energy structure in dotbracket format for the sequence in integer format"""
   a = RNA.fold_compound(seq, model)
   (mfe_structure, mfe) = a.mfe()
   subopt = return_structures_in_energy_range(0.02, seq)
   assert mfe_structure in subopt
   if len(subopt.keys()) == 1:
      return mfe_structure
   else:
      return '|'

def get_all_mfe_structures_seq_str(seq):
   """get minimum free energy structure in dotbracket format for the sequence in integer format"""
   subopt = return_structures_in_energy_range(0.02, seq)
   return [s for s in subopt.keys()]


def get_unique_mfe_structure_as_int_seq_str(seq):
   """get minimum free energy structure in dotbracket format for the sequence in integer format"""
   return dotbracket_to_int(get_unique_mfe_structure_seq_str(seq), check_size_limit=False)
###############################################################################################
## converting between structure representations
###############################################################################################
def dotbracket_to_binary(dotbracketstring):
   """translating each symbol in the dotbracket string a two-digit binary number;
   translation is defined such that the number of '1' digits is the number of paired positions;
   similar to Dingle, K., Camargo, C. Q. and Louis, A. A. Input-output maps are strongly biased towards simple outputs. Nature Communications 9, (2018)."""
   binstr = '1'
   for char in dotbracketstring:
      binstr = binstr + db_to_bin[char]
   return binstr

def dotbracket_to_int(dotbracketstring, check_size_limit=True):
   """convert dotbracket format to integer format:
   this is achieved by translating each symbol into a two-digit binary number
   and then converting this into a base-10 integer; a leading '1' is added, so that leading '0's in the binary string matter for the decimal
   if check_size_limit, the function tests whether the integer is within the range of the numpy uint32 datatype"""
   if dotbracketstring == '|': #unfolded
     dotbracketstring = '.'
   binstr = dotbracket_to_binary(dotbracketstring)
   if check_size_limit:
      assert len(binstr)<32
   integer_rep = int(binstr, 2)
   if check_size_limit:
      assert 0 < integer_rep < 4294967295 #limit of uint32
   return integer_rep


def get_dotbracket_from_int(structure_int):
   """ retrieve the full dotbracket string from the integer representation"""
   dotbracketstring = ''
   bin_to_db = {'10': '(', '00': '.', '01': ')'}
   structure_bin = bin(structure_int)[3:] # cut away '0b' and the starting 1
   assert len(structure_bin) % 2 == 0
   for indexpair in range(0, len(structure_bin), 2):
      dotbracketstring = dotbracketstring + bin_to_db[structure_bin[indexpair]+structure_bin[indexpair+1]]
   if dotbracketstring == '.':
      return '|'
   return dotbracketstring



############################################################################################################
## test
############################################################################################################
if __name__ == "__main__":
   print( 'test thermodynamic_functions.py')
   
   def get_Boltzmann_probabilities_viennarna(sequence, potential_structures_for_seq):
      """to ensure Boltzmann distribution is used correctly in ViennaRNA,
      we calculate the complete distribution for a full enumeration of secondary structures both manually, based on eval_structure
      and using ViennaRNA (this function)"""
      kbT = RNA.exp_param().kT/1000.0
      a = RNA.fold_compound(sequence, model)
      (mfe_structure, mfe) = a.mfe()
      a.exp_params_rescale(mfe)
      (prob_vector, dG) = a.pf()
      Boltzmannp = []
      for s in potential_structures_for_seq: 
         if sequence_compatible_with_basepairs(sequence, s):
            Boltzmannp.append(a.pr_structure(s))
         else:
            Boltzmannp.append(0.0)
      #print 'partition function computed ViennaRNA:', np.exp(-dG/kbT)
      return Boltzmannp

   def get_Boltzmann_probabilities_manually(sequence, potential_structures_for_seq):
      """to ensure Boltzmann distribution is used correctly in ViennaRNA,
      we calculate the complete distribution for a full enumeration of secondary structures both manually (this function), based on eval_structure
      and using ViennaRNA"""
      a = RNA.fold_compound(sequence, model)
      (mfe_structure, mfe) = a.mfe()
      a.exp_params_rescale(mfe)
      kbT = RNA.exp_param().kT/1000.0
      Boltzmann_factors = []
      for s in potential_structures_for_seq: 
         if sequence_compatible_with_basepairs(sequence, s):
            Boltzmann_factors.append(np.exp(-a.eval_structure(s)/kbT))
         else:
            Boltzmann_factors.append(0.0)
      #print 'partition function computed manually:', np.sum(Boltzmann_factors)
      return np.divide(Boltzmann_factors, np.sum(Boltzmann_factors))
   ############################################################################################################
   print('setup')
   ############################################################################################################
   import functions_fullstructures.functions_for_sampling_with_bp_swap as sample #
   all_length_fifteen_structures_no_isolated_bps = ['((((...))...)).', '..((........)).', '.....((...))...', '......((...))..', '...((....))....', 
                                                     '(((((...)).))).', '.((((....))..))', '(((((.....)))))', '...((.((...))))', '....((....))...', 
                                                     '((........))...', '...(((...)))...', '.....((......))', '((..((....)))).', '.((......))....', 
                                                     '(((((...)))))..', '..((..((...))))', '.((....))......', '......((.....))', '.(((.((...)))))', 
                                                     '..((((....)).))', '..(((((...)))))', '((.((...)).))..', '((.(((...))).))', '..((.......))..', 
                                                     '........((...))', '.((((....)).)).', '......((....)).', '((..((...))))..', '((((.....)).)).', 
                                                     '.(((......)))..', '..((.((....))))', '.((..........))', '.((..((...)))).', '..((((...))))..', 
                                                     '....((.....))..', '.((..((....))))', '(((......)))...', '((.((....))..))', '.((.((...))))..', 
                                                     '....((.......))', '((((...)).))...', '....(((...)))..', '.((.(((...)))))', '((...)).((...))', 
                                                     '...((((....))))', '((((.....))))..', '.....(((....)))', '.(((((...)).)))', '...((.....))...', 
                                                     '.((........))..', '....((......)).', '....(((....))).', '..(((...)))....', '.((..((...)).))', 
                                                     '((((...))..))..', '((...........))', '..((...))......', '....((((...))))', '.(((.......))).', 
                                                     '((((......)))).', '.((.((...)).)).', '..((.........))', '.......((...)).', '.((((......))))', 
                                                     '((.((...))...))', '....(((.....)))', '((..((....)).))', '..((((.....))))', '.(((........)))', 
                                                     '.((...))((...))', '((.((......))))', '((((...))))....', '((((....)).))..', '(((((....)).)))', 
                                                     '...((...)).....', '((.(((...))))).', '(((.((...)).)))', '.((.((.....))))', '((((((...))))))', 
                                                     '((.((.....)).))', '(((((...))..)))', '.((.((....)))).', '((.........))..', '((...))((....))', 
                                                     '((((......)).))', '(((..((...)))))', '..((....)).....', '.((((.....)).))', '...(((....)))..', 
                                                     '..((.((...)))).', '((..(((...)))))', '...((((...)).))', '((((....))...))', '...(((......)))', 
                                                     '.(((...))).....', '((....)).......', '(((....))).....', '..((((....)))).', '((..........)).', 
                                                     '((...((...)).))', '((.((...))..)).', '...((.......)).', '((.......))....', '((((....))..)).', 
                                                     '.(((((....)))))', '.((((...)).))..', '((((...))....))', '((...((...)))).', '((.((....))))..', 
                                                     '(((((...))).)).', '......(((...)))', '.......((....))', '..(((......))).', '(((.((....)))))', 
                                                     '...(((.....))).', '...((((...)))).', '((...))........', '.....((.....)).', '((...))((...)).', 
                                                     '((....((...))))', '((((....))))...', '..(((.....)))..', '((......)).....', '..((((...))..))', 
                                                     '.((.((...))..))', '((.((....)).)).', '.....((....))..', '.(((.....)))...', '.((.......))...', 
                                                     '((.(((....)))))', '(((((...)))..))', '((....))((...))', '.((((...))..)).', '.((((.....)))).', 
                                                     '((.....))......', '..(((.......)))', '.(((((...))).))', '((...((....))))', '.((.........)).', 
                                                     '.(((((...))))).', '..((.((...)).))', '...((......))..', '.....(((...))).', '....((...))....', 
                                                     '((((.......))))', '.((((...))))...', '.((...((...))))', '(((........))).', '((..((...))..))', 
                                                     '((.((...))))...', '(((((....))))).', '((..((...)).)).', '(((.....)))....', '...((........))', 
                                                     '(((.((...))))).', '((((.....))..))', '.(((....)))....', '.((.((....)).))', '..((((...)).)).', 
                                                     '.((...)).......', '..((.....))....', '..((......))...', '.((((...))...))', '((..((.....))))', 
                                                     '..(((....)))...', '.((((....))))..', '(((.........)))', '((.((.....)))).', '.((.....)).....', 
                                                     '(((...)))......', '(((((....))).))', '(((.......)))..', '...............']
   all_length_fourteen_structures_with_isolated_bps = ['..............',] + sample.read_structure_list('all_length_fourteen_structures_with_isolated_bps.txt', allow_isolated_bps=True) 
   ############################################################################################################
   print('\n-------------\n\ntest structure conversion')
   ############################################################################################################
   seq_len = 20
   for test_no in range(10**4):
      seq = ''.join(np.random.choice(['A', 'C', 'G', 'U'], seq_len, replace=True))
      mfe_struct = get_unique_mfe_structure_seq_str(seq)   
      if mfe_struct == '|':
        print('multiple mfe struuctures')
        continue
      assert get_dotbracket_from_int(dotbracket_to_int(mfe_struct, check_size_limit=False)) == mfe_struct
   ############################################################################################################
   print('\n-------------\n\ntest Boltzman frequency calculations')
   ############################################################################################################
   seq_len = 15
   G_range_kcal_test = 5
   n_tests = 10**5
   for test_no in range(n_tests):
      if test_no % 100 == 0:
         print(test_no/n_tests*100, '% of tests complete' )
      seq = ''.join(np.random.choice(['A', 'C', 'G', 'U'], seq_len, replace=True))
      mfe_struct = get_unique_mfe_structure_seq_str(seq)
      if mfe_struct == '|':
        print('multiple mfe struuctures')
        continue
      mfe_struct_index = all_length_fifteen_structures_no_isolated_bps.index(mfe_struct)
      Boltzmann_prob_list_manual =  get_Boltzmann_probabilities_manually(seq, all_length_fifteen_structures_no_isolated_bps)
      Boltzmann_prob_list = get_Boltzmann_freq_list(seq, all_length_fifteen_structures_no_isolated_bps)
      lowenergy_ensemble = get_Boltzmann_freq_dict_lowenergyrange(seq, G_range_kcal = G_range_kcal_test)
      for structureindex, structure in enumerate(all_length_fifteen_structures_no_isolated_bps):
         B1, B2 = Boltzmann_prob_list_manual[structureindex], Boltzmann_prob_list[structureindex]
         assert max(B1, B2) < 10** (-10) or abs(B1/B2 - 1) < 0.01
         if Boltzmann_prob_list_manual[structureindex]/Boltzmann_prob_list_manual[mfe_struct_index] > np.exp(-1 * G_range_kcal_test/kbT_RNA):
            assert structure in lowenergy_ensemble
            assert abs(lowenergy_ensemble[structure]/Boltzmann_prob_list[structureindex] - 1) < 0.01
   ############################################################################################################
   print('\n-------------\n\ntest Boltzman frequency calculations against Vienna when isolated base pairs allowed - then can also use ViennaRNA function')
   ############################################################################################################
   model.noLP = 0 # no isolate base pairs
   seq_len = 14
   G_range_kcal_test = 15
   n_tests = 10**5
   for test_no in range(n_tests):
      if test_no % 100 == 0:
         print(test_no/n_tests*100, '% of tests complete' )
      seq = ''.join(np.random.choice(['A', 'C', 'G', 'U'], seq_len, replace=True))
      mfe_struct = get_unique_mfe_structure_seq_str(seq)
      if mfe_struct == '|':
        print('multiple mfe struuctures')
        continue
      mfe_struct_index = all_length_fourteen_structures_with_isolated_bps.index(mfe_struct)
      Boltzmann_prob_list_manual =  get_Boltzmann_probabilities_manually(seq, all_length_fourteen_structures_with_isolated_bps)
      Boltzmann_prob_list_vienna = get_Boltzmann_probabilities_viennarna(seq, all_length_fourteen_structures_with_isolated_bps)
      lowenergy_ensemble = get_Boltzmann_freq_dict_lowenergyrange(seq, G_range_kcal = G_range_kcal_test, test_no_isolated_bp=False)
      for structureindex, structure in enumerate(all_length_fourteen_structures_with_isolated_bps):
         B1, B2 = Boltzmann_prob_list_manual[structureindex], Boltzmann_prob_list_vienna[structureindex]
         assert max(B1, B2) < 10** (-10) or abs(B1/B2 - 1) < 0.01
         if Boltzmann_prob_list_manual[structureindex]/Boltzmann_prob_list_manual[mfe_struct_index] > np.exp(-1 * G_range_kcal_test/kbT_RNA):
            assert structure in lowenergy_ensemble
            assert abs(lowenergy_ensemble[structure]/Boltzmann_prob_list_vienna[structureindex] - 1) < 0.01
   ############################################################################################################
   print( '\n-------------\n\nfinished thermodynamic functions tests\n-------------\n\n')
   ############################################################################################################









