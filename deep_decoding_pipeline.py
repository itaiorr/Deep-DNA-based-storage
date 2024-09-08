from DeepEncoderDecoder.multi_threaded_preprocessor import read_preprocessor
from DeepEncoderDecoder.binning import binner
from DeepEncoderDecoder.decoder_pipeline import main_decoder
from DNN.inference import inference_run
import torch
if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn', force=True)


# this parameter presents the lib length of 140
lib_length = 140

# path to the sequencing output - can be .txt or a .fastq file or a path to a folder with many .fastq files
# this is the file the includes all the basecalled reads that we want to processed
reads_path = "/Users/omersabary/Documents/dataNanopore-pilotSep11/full_omer_alex_hadas/pass/seq.txt"
reads_path = "./DeepEncoderDecoder/fastq_runid_f9875c0bb5171f8bae2f76aa5d9ac39431889861_1221_0.fastq"

# path (including file name) in which the trimmed reads will be stored
reads_trimmed_path="./reads_trimmed.txt"

# path the includes the design of the encoded sequences in a csv format,
# First column corresponds to the index and the second column is the full encoded sequence
design_file_path = "./DeepEncoderDecoder/deep_design.csv"

# path of the clusters
clusters_path = "./"

# path of the DNN inference results
inference_path = "./inference/"



## Step 1 - primer trimming
print("Step 1 primer trimming...")

read_preprocessor(reads_path, reads_trimmed_path, lib_length)

## step 2 - binning the reads
print("Step 2 binner...")

binner(design_file_path=design_file_path, reads_file_path=reads_trimmed_path)

## step 3 - creating the DNN inference
##
### inference from itai
print("Step 3 Inference...")
#if __name__ == '__main__':
#    inference_run()
#exit(0)

## step 4 Decoding the file from the inference
'''
This the main function of the decoder. The function gets the following arguments 
#1 Code Parameters - These parameters define the redundancy of the code as described in the paper. 
- erasure_fraction (0.04) - fraction of rows (clusters) that we assume as erased (missing/by confidence function). 
- substitution_fraction (0.0075) - fraction of clusters, with estimation that have more than 3 substitution errors. 
- almost_correct_fraction=0.016 - fraction of clusters, with estimation that have  3 or less  substitution errors.

#2 Inference path - path to the inference, given in json files. 
- inf_path = path_to_inf_results

#3 output file path 
 -     out_put_file_path = "out_decoder.bin"

#4  File identifier(s) - this parameter defines the file identifier. 
        A/C - semantic file 
        G/T - random file 
 -  file_identifier = {'A', 'C'}
 
 #5 Confidence Mode - reprsents the condifence mode of the system
conf_mode = 
 - "DNN"
 - "DNN+Conf"
 - "DNN+Conf+CPL"

'''

erasure_fraction=0.04
substitution_fraction=0.0075
almost_correct_fraction=0.016
## inference_path = "./inference/"
inference_path = "/Users/omersabary/Documents/dataNanopore-pilotSep11/inf_itai_final/full_nanopore_size_16_old_see_dnn_robustness" ### delete this one
inference_path = "./DeepEncoderDecoder/results_full/" ### delete this one

out_put_file_path = "out_decoder.bin"
file_identifier = {'G', 'T'}
conf_mode="DNN+Conf+CPL"
print("Step 4 Decoding... Mode: " +conf_mode)
main_decoder(erasure_fraction, substitution_fraction, almost_correct_fraction, inference_path,
                      out_put_file_path, file_identifier, conf_mode)
