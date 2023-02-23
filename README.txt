This folder contains the code required for the preprint "The Boltzmann distributions of folded molecular structures predict likely changes through random mutations" (Nora S Martin, Sebastian E Ahnert, bioRxiv 2023.02.22.529545; doi: https://doi.org/10.1101/2023.02.22.529545)

This code uses ViennaRNA (2.4.14) and Python 3.7, and various standard Python packages (matplotlib, pandas etc.).

References:
- Site-scanning is adapted from Weiß & Ahnert (2020); here base pair swaps are used as well as point mutations, as in our previous work (Martin and Ahnert 2022).
- ViennaRNA manual: https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/RNAlib-2.4.14.pdf
- HP model: this implementation follows the methods in Greenbury et al. (especially Sam Greenbury's PhD thesis, University of Cambridge 2014), except that the directionality of the protein chain on the lattice is not discarded in our analysis. However, this convention can be changed easily and this allowed us to test our code against the data in Greenbury et al. (Nature Ecology & Evolution 2022).
- For the conditional complexity calculations, a file KC.py is required, which implements the complexity estimates in the information-theoretic approach by Dingle et al. (2022). This can be requested from Dingle et al.
- The neutral set size estimator (Jörg et al. BMC Bioinformatics. 9, 464, 2008) can be downloaded and compiled from https://www.ieu.uzh.ch/wagner/software/RNA08/index.html . To disable isolated base pairs, one needs to set noLonelyPairs to one in the c code. We then wrote a script neutral_set_size_estimator to run it from Python.
Further details on the methods used and the underlying references can be found in the preprint.


Generic functions (for example for analysing the RNA genotype-phenotype map and creating plots) are adapted from our previous work on the RNA genotype-phenotype map.


For the RNA analysis, we need to run (in addition to the neutral set size estimator):
get_sampling_data_fullstructures_phi_plot.py
test_sitescanning_fullstructures.py

For the HP model, we need to run :
save_data_HP.py
HP_neutral_set_size_vs_Boltz.py 
get_correlations_HPmodel.py (for different values of kbT in the parameter file)
HP_different_temp_plots.py

Finally, the script plot_all.py generates the plots.