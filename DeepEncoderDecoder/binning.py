import csv
import json
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def load_csv(design_file):
    """
    Load the barcode and sequence data from the CSV file.
    """
    data = {}
    with open(design_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            index = row['barcode']
            sequence = row['sequence']
            data[index] = sequence
    return data


def load_reads(reads_file):
    """
    Load the DNA reads from the text file and bin them by the first 12 symbols.
    """
    bins = defaultdict(list)
    with open(reads_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                index = line[:12]
                bins[index].append(line)
    return bins


def process_bin(index, sequence, reads):
    """
    Process each bin, selecting up to 16 random reads if there are more than 16.
    """
    if len(reads) > 16:
        reads = random.sample(reads, 16)

    bin_data = {
        "index": index,
        "data": sequence,
        "noisy_copies": reads
    }

    # Ensure the directory exists
    os.makedirs("./clusters", exist_ok=True)

    json_filename = os.path.join("./clusters", f"{index}.json")
    with open(json_filename, 'w') as json_file:
        json.dump(bin_data, json_file, indent=4)


def create_json_files_parallel(binned_reads, barcode_data):
    """
    Create JSON files for each barcode with the corresponding sequences and noisy copies in parallel.
    """
    with ThreadPoolExecutor() as executor:
        futures = []
        for index, sequence in barcode_data.items():
            if index in binned_reads:
                reads = binned_reads[index]
                futures.append(executor.submit(process_bin, index, sequence, reads))

        # Progress bar for tracking task completion
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing bins"):
            future.result()  # Ensure all futures are completed


def binner(design_file_path = 'deep_design.csv', reads_file_path = 'reads_trimmed.txt'):

    index_design = load_csv(design_file_path)
    binned_reads = load_reads(reads_file_path)

    create_json_files_parallel(binned_reads, index_design)



