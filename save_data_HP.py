#!/usr/bin/env python3
import numpy as np
from copy import deepcopy
from HP_model.HPfunctions import *
from os.path import isfile
from multiprocessing import Pool
from functools import partial
import pandas as pd
import parameters_HP as param



def number_to_int_seq(n, K, L):
   seq = []
   for i in range(L):
      seq.append(n//K**(L-i -1))
      n = n%K**(L-i -1)
   return seq

def fold_mfe_from_int(n, K, L, contact_map_list):
   #print('finished one structure')
   return find_mfe(number_to_int_seq(n, K, L), contact_map_list)

###############################################################################################
print( 'sequence to mfe structure' )
###############################################################################################
if not isfile(param.GPmap_filename):
   mfe_function = partial(fold_mfe_from_int, contact_map_list=param.contact_map_list, K=param.K, L= param.L)
   with Pool(25) as p:
      mfestructurelist = p.map(mfe_function, np.arange(param.K**param.L))
   print('finished parallel', flush=True)
   GPmap =  np.zeros((param.K,)*param.L, dtype='uint32')
   assert len(param.contact_map_list) + 2 < 2**32
   structureindex = 0
   for g in np.ndindex(GPmap.shape):
      GPmap[tuple(g)] = mfestructurelist[structureindex]
      assert tuple(g) == tuple(number_to_int_seq(structureindex, K=param.K, L= param.L))
      structureindex += 1
   print('number of phenotypes (incl unfolded), ', np.unique(GPmap))
   print('number of genotypes', param.K**param.L)
   np.save(param.GPmap_filename, GPmap, allow_pickle=False)
   del mfestructurelist, GPmap
###############################################################################################
print( 'test a few random sequences' )
###############################################################################################
GPmap = np.load(param.GPmap_filename)
for i in range(10**3):
   seq = tuple(list(np.random.choice([1, 0], param.L, replace=True)))
   assert GPmap[seq] == find_mfe(seq, param.contact_map_list)

   
