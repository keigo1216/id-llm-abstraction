The corpora directory contains the sequences we used as input for intrinsic dimensionality (ID) and information imbalance (II) computation. They consist of samples from the original corpora we used, and from their shuffled versions. The sequences consist of 20 tokens, but the files also include a 21st, tab-delimited token, for surprisal computation.

We extracted last-token representations for these sequences as produced by the following HuggingFace models:

- meta-llama/Meta-Llama-3-8B
- mistralai/Mistral-7B-v0.1
- allenai/OLMo-7B
- facebook/opt-6.7b
- EleutherAI/pythia-6.9b

Representations were extracted using the `extract_final_representations.py` script. For example, the Llama sane Bookcorpus-based representations were extracted with the following command:

`python scripts/extract_final_representations.py meta-llama/Meta-Llama-3-8B 1 corpora/bookcorpus_sane_ds.txt reps_pickles/hidden_bookcorpus_sane_llama_reps`

where the output representations are stored in the `reps_pickles` directory in a pickle file named `hidden_bookcorpus_sane_llama_reps.pickle`.


## ID COMPUTATION

Given the representation pickles, we compute the ID using GRIDE (`scripts/dadapy_id.py`) at scales 2**1 -> 2**13.
The `dadapy_id.py` code is hard-coded to split the representations into 5 equally-sized partitions (the same as the ones for II computation below). For each partition, it saves the ID computation for each (layer, model, dataset, GRIDE scale) combo.

Later, the GRIDE scale is chosen by plotting the ID with respect to each scale setting, for each layer, model, and dataset (plotting code not included).


Note that ID and II computation requires installing the DADApy package (https://github.com/sissa-data-science/DADApy).


## INFORMATION IMBALANCE COMPUTATION


Given the representation pickles created as described above, we computed II in two phases.

We first use `precompute_distances.py` to create data-structures containing, for each representation on each layer, a list of indices to all other representations on the same layer ordered by Euclidean distance. For example, for the Llama Bookcorpus-based representations created as described above, we used the following command:

`python scripts/precompute_distances.py reps_pickles/hidden_bookcorpus_sane_llama_reps.pickle distance_indices/bookcorpus_llama`

The `precompute_distances.py` script is hard-coded to split the representations into 5 equally-sized partitions. For each of these partitions, it creates a separate pickle file with the representation-by-representation distance information for each layer. For example, the output pickle file `distance_indices/bookcorpus_llama_4_22_indices.pickle` contains the indices for layer 22 given the 4th partition of the data.

These pickle files are then used by the script `information_imbalance.py` to calcuate the information imbalance across the layers of the same or different models. For example, to compute layer-by-layer information imbalance scores between Llama and Mistral, given partition 2 of the Bookcorpus and the files created as above, we run:

`python scripts/information_imbalance.py --model1 distance_indices/bookcorpus_llama_2_ --model2 distance_indices/bookcorpus_mistral_2_ --target_file model_comparison_results/bookcorpus_llama_mistral_2 --k 1`

This will leave two text files in the `model_comparison_results` directory. The `bookcorpus_llama_mistral_1_ab.dat` file contains the matrix of layer-by-layer information imbalances from Llama to Mistral, and the `bookcorpus_llama_mistral_1_ba.dat` the transpose of the one from Mistral to Llama. Note that layer 0 is not included in these matrices.

To get within-model imbalances, we simply pass the same model to the `--model1` and `--model2` arguments.
