# DNAformer allows Scalable and Robust DNA-based Storage via Coding Theory and Deep Learning 


This repository includes the methods that were used in the work. 

![Example Image](pipeline_(fig1).png)
The figure was created with BioRender.com. 

### Link to our datasets will be shared upon request. 

## The repository includes the following folders. 

1. CPL - Implementation of the CPL algorithm.
2. DNN - Implementation of the DNN. 
3. data_generator - Implementation of the simulated data generator that was used to train the DNN. 
4. Encoder_Decoder - Implemenation of our encoding and decoding algorithms.  
5. Data Utilities - Scripts that are used to parse and read our data.

## Required Python Packages

The encoder and decoder are implemented in python 3.8 and requires the following packages. 
```bash

subprocess
galois
os
math
random
numpy
pickle
tqdm 
json
os
torch
einops
sklearn
warnings
glob


```

## Full End-to-End Retrieval Pipeline

To run the end-to-end retrieval pipeline, please use the script 
Deep_decoding_pipeline.py  that can be found in this folder.

This three step should be done before running the scripts.
1. Please install the required Python packages (see above) and follow the instructions in this file (section "Required Python Packages").
2. Please locate the data (the .fastq files that are obtained from sequencing) in the folder that can be found in: "./DeepEncoderDecoder/data/". This folder currently contains a sample of three .fastq files with 8966 reads.
3. Please compile the CPL algorithm by running 'make' command in the folder "./DeepEncoderDecoder/CPL_Deep/"  (see instructions below - section "CPL Algorithm").
4. Download the trained DNN from this [link](https://drive.google.com/drive/folders/1y3cJ3bJdRcrzmEpzlKIosl-b8N5NXEWJ?usp=sharing) and place the file DNAFormer_siamese.pth in a folder named checkpoints. Place the folder named checkpoints in the main folder of this repository.

```bash
python3 deep_decoding_pipeline.py
```

The full decoding pipeline includes the following components: 
1. Preprocessing of the reads—This includes primer trimming and preprocessing of the reads obtained from sequencing. 
This step trims the unnecessary parts of the primers from the reads before passing them through the decoding pipeline. 
2. Binning algorithm - this algorithm is performed on the reads to create the clusters.
This step bins the obtained reads based on the indices, and the binned reads are later used as inputs the DNAformer. 
3. DNAformer - this step includes creating the inference of the DNAformer, including the margin safety mechanism. 
In this step DNAformer is used to estimate the encoded sequences from the obtained reads. 
4. Decoding of the information. 
This step is used to decode the information from the DNN inference. 



Full encoding pipeline is given in the script encode.py in Encoder_Decoder folder (as described below). 
```bash
python3 encode.py
```

Error simulation can be done with external tools, for example: 
MESA simulator: 
Schwarz, Michael and Welzel, Marius and Kabdullayeva, Tolganay and Becker, Anke and Freisleben, Bernd and Heider, Dominik, “MESA: automated assessment of synthetic DNA fragments and simulation of DNA synthesis, storage, sequencing and PCR errors,” Bioinformatics, vol. 36, no. 11, pp. 3322–3326, 2020. LINK: https://mesa.mosla.de/[https://mesa.mosla.de/] .

DeepSimulator: 
Li, Yu and Han, Renmin and Bi, Chongwei and Li, Mo and Wang, Sheng and Gao, Xin, “DeepSimulator: a deep simulator for Nanopore sequencing,” Bioinformatics, vol. 34, no. 17, pp. 2899–2908, 2018. LINK: https://github.com/liyu95/DeepSimulator[https://github.com/liyu95/DeepSimulator].

## CPL algorithm
![cpl_pic](cpl.png)
The CPL algorithm is implemented in c++. 
Installation of the g++ compiler is required (see link: https://gcc.gnu.org/). 


### CPL algorithm - Compilation

Compilation is highly recommended by running the makefile command. 

```bash
make
```

Alternatively, compilation can be done by running the following command. 

```bash
g++ -std=c++0x -O3 -g3 -Wall -c -fmessage-length=0 -o *.cpp g++ -o main *.o
```


### Running the CPL Algorithm
To use the algorithm, it is required to bin the reads and formatting them according to the following format:

Each cluster of reads appears in the file with a header followed by the reads. More specifically:
1. The header consists of 2 lines, the first corresponds to the encoded sequence of the clusters (if the encoded sequence is unknown then the line should be emtpy line), and the second is a line of 18x“*” that should be ignored
2. The reads in the clusters are provided after the header, where each read is given in a separate line
3. Each cluster is ended with two empty lines


```bash

./main path_to_binned_file/binnedfile.txt path_to_results/ >results.txt

```

### Encoder/Decoder - Usage

![image](encoding.png)

Encoding the information should be done by running the encode command in encoder.py. 
The default parameters are the one that we used in our pipeline, however different redundancy can be applied by changing the code pararmeters. 
```bash
python3 encode.py
```
The output of the encoder will appear in the file data_with_indices.txt, where each line corresponds to encoded sequence with its index. 

Decoding should be done by running the decode command in decoder.py. 
(Make sure editing the path to the inference results made by the DNN and the cpl results made by the CPL)
```bash
python3 decode.py
```

The matrix H that was used throughout our encoding process can be found in H_matrix_identity.py.

The Reed Solomon implementation was done using the schifra library [link](https://www.schifra.com/).
Part of the encoder-decoder was implemented by Dvir Ben-Shabat.
