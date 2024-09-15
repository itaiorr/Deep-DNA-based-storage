from error_rates_setup import *
from cluster_generator import *
import json

if __name__ == '__main__':
    # options are 'Ilumina NextSeq', 'Ilumina miSeq' and 'MinION'
    sequencing_tech = 'Ilumina NextSeq'
    # options are 'Twist Bioscience', 'CustomArray' and 'IDT'
    synthesis_tech = 'Twist Bioscience'
    
    # get tech. error rates
    errors_prob = ErrorRates()
    errors_prob.set_values(sequencing_tech, synthesis_tech)

    # generate cluster
    label = 'CGCGTTGGCACGGAGCGAACCATGCGAGGTTTGGCTCAGGTGATATCGAGAAAATTCTAATACGAATTTGAGATCCCTAGAGCCG' \
            'TTTAAATGCCTATAATAGGTGCCAACGTTGCCCTAAGAATCTGGCATTGTGGAGGCCAATCTCCTCCCGTATATCTTGACGATCC' \
            'TCTCTCAGTTTTGAGTTAACTACAGGTTAGG'
    generator = ClusterGenerator(errors_prob.general_errors, errors_prob.per_base_errors, label)
    generator.generate_cluster(0.5)
    cluster_map = {'label': label, 'noisy_copies': generator.copies}

    # next line converts to json (if needed)
    # cluster_json = json.dumps(cluster_map)
