import networkx as nx

############################################################################################################
## manipulate structures
############################################################################################################

def dotbracket_to_coarsegrained(dotbracketstring):
   """transform a full dotbracket representation to a coarse-grained 
   (type 1 as defined in Janssen, Reeder, Giegerich (2008). BMC bioinformatics, https://doi.org/10.1186/1471-2105-9-131)"""
   fine_grained_to_coarse_grained_symbol = {'(': '[', ')': ']', '.': '_'}
   basepair_index_mapping = get_basepair_indices_from_dotbracket(dotbracketstring)
   coarse_grained_string = ''
   for charindex, char in enumerate(dotbracketstring):
      if charindex == 0  or dotbracketstring[charindex-1] != dotbracketstring[charindex]:
         coarse_grained_string = coarse_grained_string + fine_grained_to_coarse_grained_symbol[char]
      elif dotbracketstring[charindex-1] == dotbracketstring[charindex] and dotbracketstring[charindex] != '.': #two subsequent brackets of same type
         if not abs(basepair_index_mapping[charindex]-basepair_index_mapping[charindex-1])<1.5:
            coarse_grained_string = coarse_grained_string + fine_grained_to_coarse_grained_symbol[char]
         else:
            pass
      else:
         pass
   return coarse_grained_string


############################################################################################################
## find lengths of structural elements 
############################################################################################################
def has_length_one_stack(dotbracketstring):
   """test if dotbracketstring has isolated base pairs"""
   for pos, char in enumerate(dotbracketstring):
      if char in [')', '('] and find_len_of_stack(pos, dotbracketstring) < 2:
         return 1
   return 0

def find_len_of_stack(pos, dotbracketstring):
   """ return the length of the stack at position pos in the structure given by the dot-bracket string
   bulges are defined as the end of the stack on both strands, i.e. if a base pair is at i, j, the base pair at i+1, j-2 would not belong to the same stack"""
   basepair_index_mapping = get_basepair_indices_from_dotbracket(dotbracketstring)
   assert pos in basepair_index_mapping
   node_of_basepair = min(pos, basepair_index_mapping[pos])
   ## make network of basepairs and connect adjacent basepairs in stacks - then stack size is size of component
   base_pair_neighbour_graph = nx.Graph()
   base_pair_neighbour_graph.add_nodes_from(set([min(b) for b in basepair_index_mapping.items()])) # each base pair is represented by the pos of the opening bracket
   for b1, b2 in basepair_index_mapping.items():
      for a1, a2 in basepair_index_mapping.items():
         if b1<b2 and a1<a2: # both ordered from ( to )
            if b1 != a1: # distinct bas pairs
               if abs(b1-a1) == 1 and abs(b2-a2) == 1:
                  base_pair_neighbour_graph.add_edge(a1, b1)
   return len(nx.node_connected_component(base_pair_neighbour_graph, node_of_basepair))


############################################################################################################
## extract base pairs
############################################################################################################

def get_basepair_indices_from_dotbracket(dotbracketstring):
   """extract a dictionary mapping each paired position with its partner:
   each base pair is represented twice: mapping from opening to closing bracket and vice versa"""
   assert '[' not in dotbracketstring
   base_pair_mapping = {}
   number_open_brackets = 0
   opening_level_vs_index = {}
   for charindex, char in enumerate(dotbracketstring):
      if char == '(':
         number_open_brackets += 1
         opening_level_vs_index[number_open_brackets] = charindex
      elif char == ')':
         base_pair_mapping[charindex] = opening_level_vs_index[number_open_brackets]
         base_pair_mapping[opening_level_vs_index[number_open_brackets]] = charindex
         del opening_level_vs_index[number_open_brackets]
         number_open_brackets -= 1
      elif char == '.':
         pass
      else:
         raise ValueError('invalid character in dot-bracket string', dotbracketstring)
      if number_open_brackets < 0:
         raise ValueError('invalid dot-bracket string')
   if number_open_brackets != 0:
      raise ValueError('invalid dot-bracket string')
   return base_pair_mapping



############################################################################################################
## check viability of structure
############################################################################################################

def sequence_compatible_with_basepairs(sequence, structure):
   """check if the input sequence (string containing AUGC) is 
   compatibale with the dot-bracket input structure,
   i.e. paired sites are a Watson-Crick pair or GU"""
   allowed_basepairs = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG']
   for b in sequence:
      assert b in ['A', 'U', 'G', 'C']
   bp_mapping = get_basepair_indices_from_dotbracket(structure)
   for baseindex1, baseindex2 in bp_mapping.items():
      if sequence[baseindex1]+sequence[baseindex2] not in allowed_basepairs:
         return False
   return True

def hairpin_loops_long_enough(structure):
   """check if any paired sites in the dot-bracket input structure
   are at least four sites apart"""
   bp_mapping = get_basepair_indices_from_dotbracket(structure)
   for baseindex1, baseindex2 in bp_mapping.items():
      if abs(baseindex2-baseindex1) < 4:
         return False
   return True



def is_likely_to_be_valid_structure(structure, allow_isolated_bps=False):
   """tests if a structure in dotbracket format is likely to be a valid structure:
   basepairs closed, length of hairpin loops (>3), presence of basepairs and optionally isolated base pairs"""
   if not basepairs_closed(structure):
      return False
   if not hairpin_loops_long_enough(structure):
      return False
   if not structure.count(')') > 0:
      return False
   if not allow_isolated_bps and has_length_one_stack(structure):
      return False
   else:
      return True


def basepairs_closed(structure):
   """test if all brackets are closed correctly in a dot-bracket string"""
   try:
      bp_map = get_basepair_indices_from_dotbracket(structure)
      return True
   except (ValueError, KeyError):
      return False

############################################################################################################
## test
############################################################################################################
if __name__ == "__main__":
   print( 'test: rna_structural_functions.py')
   teststructure1 = '...(((..((...)))))..((....)).'
   teststructure2 = '(((...))).'
   ####
   print( 'test stack of length one')
   ####
   assert has_length_one_stack('.((.(...)))..') and has_length_one_stack('.((.(...).))..') and has_length_one_stack('.((.(...).))..')
   assert has_length_one_stack('.(...)...')
   assert not (has_length_one_stack(teststructure1) or has_length_one_stack(teststructure2))
   ####
   print( 'convert to coarsegrained structure')
   ####
   assert dotbracket_to_coarsegrained(teststructure1) == '_[_[_]]_[_]_'
   assert dotbracket_to_coarsegrained(teststructure2) == '[_]_'
   assert dotbracket_to_coarsegrained('.((.(...)))..') == '_[_[_]]_' and dotbracket_to_coarsegrained('.((.(...).))..') == '_[_[_]_]_' and dotbracket_to_coarsegrained('.(((...).))..') == '_[[_]_]_'
   ####
   print( 'test stack lengths')
   ####
   assert find_len_of_stack(3, teststructure1) == find_len_of_stack(4, teststructure1) == find_len_of_stack(5, teststructure1) == find_len_of_stack(15, teststructure1) == find_len_of_stack(16, teststructure1) == 3
   assert find_len_of_stack(8, teststructure1) == find_len_of_stack(9, teststructure1) == find_len_of_stack(13, teststructure1) == find_len_of_stack(14, teststructure1) == 2
   assert find_len_of_stack(0, teststructure2) == find_len_of_stack(6, teststructure2) == find_len_of_stack(7, teststructure2) == find_len_of_stack(8, teststructure2) == 3
   ####
   print( 'test base pair extraction')
   ####
   bp_mapping1 = get_basepair_indices_from_dotbracket(teststructure1)
   assert bp_mapping1[3] == 17 and bp_mapping1[4] == 16 and bp_mapping1[5] == 15 and bp_mapping1[8] == 14 and bp_mapping1[20] == 27
   assert bp_mapping1[17] == 3 and bp_mapping1[16] == 4 and bp_mapping1[15] == 5 and bp_mapping1[14] == 8 and bp_mapping1[27] == 20
   bp_mapping2 = get_basepair_indices_from_dotbracket(teststructure2)
   assert bp_mapping2[0] == 8 and bp_mapping2[1] == 7 and bp_mapping2[2] == 6 

   ####
   print( 'test sequence-structure compatibility')
   ####
   teststructure1 = '...(((..((...)))))..((....)).'
   teststructure2 = '(((...))).'
   assert sequence_compatible_with_basepairs('AAGGGCCAGGAUUCCGCCGAGCGAUAGCA', teststructure1)
   assert not sequence_compatible_with_basepairs('AAGUGCCAGGAUUCCGCCGAGCGAUAGCA', teststructure1)
   assert sequence_compatible_with_basepairs('AAGGGCAAGGAUUCCGCCGAGCGAUAGCA', teststructure1)
   assert sequence_compatible_with_basepairs('GUACGUUGC', teststructure2)
   assert not sequence_compatible_with_basepairs('GUACGUGGC', teststructure2)
   assert not sequence_compatible_with_basepairs('GUGCGUGGC', teststructure2)
   assert sequence_compatible_with_basepairs('GUACAUUGU', teststructure2)
   ####
   print( 'test length of hairpin loops')
   ####
   assert hairpin_loops_long_enough(teststructure1) and hairpin_loops_long_enough(teststructure2)
   assert not hairpin_loops_long_enough('...(((..((..)))))..((....)).')
   ####
   print( 'test whether unclosed bps are recognised correctly')
   ####
   assert basepairs_closed(teststructure1) and basepairs_closed(teststructure2) and basepairs_closed(teststructure1[1:]) and basepairs_closed(teststructure1[:-1])
   assert not basepairs_closed(teststructure2[1:])
   assert not basepairs_closed(teststructure1[:-2])
   ####
   print( 'test whether invalid structures are recognised correctly')
   ####
   assert is_likely_to_be_valid_structure('...((..((...))))..((...))..', allow_isolated_bps=False)
   assert is_likely_to_be_valid_structure('...((..((...))))..(...)..', allow_isolated_bps=True) and not is_likely_to_be_valid_structure('...((..((...))))..(...)..', allow_isolated_bps=False)
   assert not is_likely_to_be_valid_structure('...((..((....)))))..((...)..', allow_isolated_bps=False)
   assert not is_likely_to_be_valid_structure('...((..((..))))..((...))..', allow_isolated_bps=False)
   assert not is_likely_to_be_valid_structure('...(((..((...))))..((...))..', allow_isolated_bps=False)
   print( '\n\n-------------\n\n'   )

