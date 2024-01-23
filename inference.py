from imports import *
import utils as utils

#####################################################################
# Set deterministic conditions
seed = 1 
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

#####################################################################
# Load model
from model_dnaFormer_siamese import net

class config:

    #####################################################################
    # Define data generator
    synthesis_tech  = 'Twist Bioscience' 
    sequencing_tech = 'MinIONShort'         
    read_padding    = 'end'      # [end ,symmetric]
    filter_index    = True
    model_config    = 'siamese'  # [single, siamese]

    # Real data:
    # full_nanopore cluster: 109,944   label: 128   dev: 4    index: 12  sequencing: Nanopore-0922         synthesis: Twist Bioscience-0922
    # full_illumina cluster: 109,944   label: 128   dev: 4    index: 12  sequencing: Ilumina miSeq-0922    synthesis: Twist Bioscience-0922

    # pilot_illumina: cluster: 1,000   label: 128   dev: 4    index: 12  sequencing: Ilumina miSeq    synthesis: Twist Bioscience
    # pilot_nanopore: cluster: 1,000   label: 128   dev: 10   index: 12  sequencing: MinIONShort      synthesis: Twist Bioscience

    # Open source datasets:
    # Erlich:   cluster: 72,000   label: 152   dev: 10   index: 16  sequencing: Ilumina miSeq    synthesis: Twist Bioscience
    # Grass:    cluster: 4989     label: 117   dev: 11   index: 13  sequencing: Ilumina miSeq    synthesis: CustomArray
    # Luis:     cluster: 596,499  label: 110   dev: 5    index: 33  sequencing: Ilumina NextSeq  synthesis: Twist Bioscience
    # Pfitser:  cluster: 9984     label: 110   dev: 5    index: 4   sequencing: MinION           synthesis: Twist Bioscience

    #####################################################################
    # Define test dataset
    test_dataset   = 'Nanopore_single_flowcell' 

    #####################################################################
    # Data parameters
    index_length           = 12
    label_length           = 140
    corrupt_max_deviation  = 4
    noisy_copies_length    = label_length+corrupt_max_deviation 
    if filter_index:
      noisy_copies_length -= index_length
    min_number_per_cluster = 1
    max_number_per_cluster = 16
    generate_data_noise    = 0.1    # [0-1] std from nominal value
    max_false_copies       = 2      # max number of copies inserted to cluster
    false_copies_prob      = 0.3
    min_cluster_size_for_false_copies = 4
    partial_flag           = False
    noise_coef = {}
    noise_coef['del_mult'] = 1
    noise_coef['ins_mult'] = 1
    noise_coef['sub_mult'] = 1

    #####################################################################
    # DNAFormer parameters
    n_head             = 32
    activation         = 'gelu'
    num_layers         = 12
    d_model            = 1024
    alignement_filters = 128
    dim_feedforward    = 2048
    output_ch          = 4
    enc_filters        = 4
    p_dropout          = 0
    class_token        = 0

    #####################################################################
    # Define training parameters
    train_date       = ''
    model_type       = ''
    frames_per_epoch = {'inf':120_000} 
    batch_size_inf   = 100      # batch size for inference                               
    nThreads         = 12       # number of threads for data loader to use
    gpus_list        = [0]      

    #####################################################################
    # Set paths 
    base_path       = ''
    data_path       = ''
    data_csv_path   = ''
    pretrained_path = ''

#####################################################################
# Load model
model = config.net(config)
utils.printNetwork(model)

#####################################################################
# Run training
utils.run_inference(config, model)
