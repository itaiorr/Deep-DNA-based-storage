from .imports import *

###########################################################################################
# Define data loader
class DatasetFromFolder(data.Dataset):  
    def __init__(self, config, df, data_source):
        
        self.config      = config
        self.data_source = data_source
        self.filenames   = df
        self.frame_path  = 'none'

    def __len__(self):
        return len(self.filenames)
   
    def __getitem__(self, idx):
        try:
            if self.data_source=='load':
                    sample = read_data(self, idx)

            elif self.data_source=='gen_sim':
                    sample = generate_data(self, idx)
    
            return sample
        except Exception as e:
            print(f"Error in worker {os.getpid()}: {e}")
            raise e

###########################################################################################
def get_random_seq(self):
    
    label_length = self.config.label_length
    
    if self.config.filter_index:
        label_length -= self.config.index_length  
        
    return torch.randint(0,4,size=(label_length,))
    
###########################################################################################
def pad_seq(config, one_hot_noisy_copy):
    
    pad = config.noisy_copies_length - one_hot_noisy_copy.shape[-1]

    if config.read_padding=='end':
        pad_start = 0
        pad_end = pad
    elif config.read_padding=='symmetric':
        pad_start = np.floor(pad/2).astype(int)
        pad_end   = np.ceil(pad/2).astype(int)

    one_hot_noisy_copy = F.pad(one_hot_noisy_copy, (pad_start, pad_end), "constant", 0)
     
    return one_hot_noisy_copy

###########################################################################################
def one_hot_encoding(read):
         
    read = torch.tensor([ord(c) for c in read])

    read[read==65]=0 # A
    read[read==67]=1 # C
    read[read==71]=2 # G
    read[read==84]=3 # T

    return F.one_hot(read,4).transpose(0,1)
    
###########################################################################################
def num2dna(seq):
    dictt = {0:'A',1:'C',2:'G',3:'T'}

    arr = []
    for i in seq:
        for j in dictt.keys():
            if j==i:
                arr.append(dictt[j])
    label = ''.join(arr)
    
    return label
    
###########################################################################################
def generate_data(self, idx):
    
    data = {}
    
    # Generate random label
    label = num2dna(get_random_seq(self))

    # Define data denerator
    min_copies = self.config.min_number_per_cluster
    max_copies = torch.randint(self.config.min_number_per_cluster,self.config.max_number_per_cluster-1,size=(1,)).item()
        
    # Define synthesis and sequncing technologies 
    errors_prob = self.config.error_rates_setup.ErrorRates()
    errors_prob.set_values(self.config.sequencing_tech, self.config.synthesis_tech, self.config.partial_flag, self.config.noise_coef)
  
    generator = self.config.cluster_generator.ClusterGenerator(errors_prob.general_errors, errors_prob.per_base_errors, label, min_copies, max_copies)

    # Generate noisy copies
    generator.generate_cluster(delta=self.config.generate_data_noise)
    data['noisy_copies'] = generator.copies
       
    # Generate false copies in cluster
    data['noisy_copies'] = get_false_copies(self.config, data['noisy_copies'])
    data['data'] = label
    
    # Get model input (one_hot encoding)
    model_input, model_input_right, noisy_copy_length, num_noisy_copies = grab_model_input(self, data)
    noisy_copy_length = torch.tensor(noisy_copy_length)

    # Torch label
    label = one_hot_encoding(label).contiguous().int()

    # Place holders
    false_cluster = False
    index         = 'None'
    cluster_path  = 'None'
    
    # Get noisy copies
    num_noisy_copies = len(data['noisy_copies'])
    if num_noisy_copies<self.config.max_number_per_cluster:
        noisy_copies = data['noisy_copies'].copy()
        for idy in range(self.config.max_number_per_cluster- num_noisy_copies):
            noisy_copies.append('none')
    else:
        noisy_copies = data['noisy_copies'][:self.config.max_number_per_cluster]
             
    # Build sample
    sample = {'model_input':model_input,
              'model_input_right':model_input_right,
              'noisy_copies':noisy_copies, 
              'num_noisy_copies':num_noisy_copies,
              'noisy_copy_length':noisy_copy_length,
              'label':label,
              'false_cluster':false_cluster,
              'index':index,
              'cluster_path':cluster_path}
    
    return sample

###########################################################################################
def get_false_copies(config, noisy_copies):
    
    if np.random.random()<config.false_copies_prob and len(noisy_copies)>config.min_cluster_size_for_false_copies:
        num_flase_copies = np.random.randint(1,config.max_false_copies+1)
    
        # Draw random noisy copies
        index_copies = np.random.randint(len(noisy_copies), size=(num_flase_copies,))

        # Replace noisy copies with false copies   
        for i in range(num_flase_copies):
            noisy_copies[index_copies[i]] = num2dna(np.random.randint(4, size=(len(noisy_copies[index_copies[i]]),)))
        
    return noisy_copies

###########################################################################################
def read_data(self, idx):
    
    while True:

        if self.frame_path == 'none':
            cluster_path = self.filenames.loc[idx,'path']
        else:
            cluster_path = self.frame_path
            
        # Read json
        try:
            with open(cluster_path) as json_file:
                data = json.load(json_file) 
        except:
            print('Missing data:',cluster_path)
            idx = np.random.randint(len(self.filenames))
            continue
          
        # Get index
        if 'index' in data.keys():
            index = data['index']
        else:
            index = data['data'][:config.index_length]
            
        # Remove index
        if self.config.filter_index:
          data['data'] = data['data'][self.config.index_length:]
          
          for i in range(len(data['noisy_copies'])):
            data['noisy_copies'][i] = data['noisy_copies'][i][self.config.index_length:] 

        # Shuffle reads
        random.shuffle(data['noisy_copies'], random.random)

        # Filter data
        data, false_cluster, small_cluster, bad_label = filter_data(self, data)
        
        if small_cluster or bad_label or false_cluster:
            idx = np.random.randint(len(self.filenames))
            continue
            
        # Get model input (one_hot encoding)
        model_input, model_input_right, noisy_copy_length, num_noisy_copies = grab_model_input(self, data)
        noisy_copy_length = torch.tensor(noisy_copy_length)
        
        # Get label
        if false_cluster:
            label_length = self.config.label_length
            if self.config.filter_index:
              label_length -= self.config.index_length
            label = torch.zeros(4,label_length,dtype=torch.int)
        else:
            label = one_hot_encoding(data['data']).contiguous().int()
        
        # Get noisy copies
        num_noisy_copies = len(data['noisy_copies'])
        if num_noisy_copies<self.config.max_number_per_cluster:
            noisy_copies = data['noisy_copies'].copy()
            for idy in range(self.config.max_number_per_cluster- num_noisy_copies):
                noisy_copies.append('none')
        else:
            noisy_copies = data['noisy_copies'][:self.config.max_number_per_cluster]
            
                    
        break # collect sample
        
    # Build sample
    sample = {'model_input':model_input, 
              'model_input_right':model_input_right,
              'noisy_copies':noisy_copies, 
              'num_noisy_copies':num_noisy_copies,
              'noisy_copy_length':noisy_copy_length,
              'label':label,
              'false_cluster':false_cluster,
              'index':index,
              'cluster_path':cluster_path}
            
    return sample


###########################################################################################
def filter_data(self, data):
    
    # Check if false cluster
    if data['data'] == 'None':
        false_cluster = True
        small_cluster = True
        bad_label     = True
        return data, false_cluster, small_cluster, bad_label
    
    # Filter corrupt reads
    temp = len(data['noisy_copies'])
    data, small_cluster = filter_corrupt_reads(self.config, data)

    # Filter bad copies
    data['noisy_copies'] = get_filtered_copies(self.config, data)

    # Filter number of reads
    data = filter_max_read_num(self.config, data)
    
    # Check again if false cluster
    if data['data'] == 'None':
        false_cluster = True
        bad_label = False
        
    else:
        false_cluster = False
        label_length = self.config.label_length
        if self.config.filter_index:
            label_length -= self.config.index_length
        if len(data['data']) != label_length:
            bad_label = True
        else:
            bad_label = False
    
    # Return data
    return data, false_cluster, small_cluster, bad_label

###########################################################################################
def get_filtered_copies(config, cluster_dict):
    
    # Sample random batch from large clusters
    if len(cluster_dict['noisy_copies']) > config.max_number_per_cluster:
        cluster_dict['noisy_copies'] = random.sample(cluster_dict['noisy_copies'], config.max_number_per_cluster)

    # Create filtered copies list
    filtered_copies = cluster_dict['noisy_copies']

    """
    # Get edit to other cluster members
    edit_cluster = {}
    for i in range(len(cluster_dict['noisy_copies'])):
        edit_cluster[i] = []
        for j in range(len(cluster_dict['noisy_copies'])):
            if j!=i:
                edit_dist = editdistance.eval(cluster_dict['noisy_copies'][i],cluster_dict['noisy_copies'][j])
                edit_cluster[i].append(edit_dist)

        # Average edit distances
        avg_dist = sum(edit_cluster[i]) / len(edit_cluster)

        # Filter bad copy
        if avg_dist<config.noisy_copy_threshold:
            filtered_copies.append(cluster_dict['noisy_copies'][i]) 
        """
    return filtered_copies

###########################################################################################
def filter_max_read_num(config, data):
        
    if len(data['noisy_copies']) > (config.max_number_per_cluster-1):
        rng = default_rng()
        read_idx = rng.choice(len(data['noisy_copies']), size=config.max_number_per_cluster, replace=False)
        
    else:
        read_idx = np.arange(config.max_number_per_cluster)
    
    good_reads = []
    for idx, noisy_copy in enumerate(data['noisy_copies']):
        
        if idx in read_idx:
            good_reads.append(noisy_copy)
        
    data['noisy_copies'] = good_reads
    
    return data

###########################################################################################
def filter_corrupt_reads(config, data):
    good_reads = []

    label_length = config.label_length
    if config.filter_index:
        label_length -= config.index_length
        
    # Set max deviation 
    max_deviation = config.corrupt_max_deviation
    
    for idx, noisy_copy in enumerate(data['noisy_copies']):
    
        # Check copy length
        if (np.abs(len(noisy_copy) - label_length) <= max_deviation):
            good_length = True
        else:
            good_length = False
            
        # Check unique characters
        if len(set(noisy_copy)) > 4:
            corrupt_ok = False
        else:
            corrupt_ok = True      
           
        # Add to list if valid   
        if good_length and corrupt_ok:
            good_reads.append(noisy_copy)
            
    data['noisy_copies'] = good_reads
    
    if len(data['noisy_copies']) < config.min_number_per_cluster:
        small_cluster = True
    else:
        small_cluster = False

    return data, small_cluster

###########################################################################################
def grab_model_input(self, data):
    
    noisy_copies = data['noisy_copies']
    num_noisy_copies = len(data['noisy_copies'])
     
    model_input      = torch.zeros([self.config.max_number_per_cluster,4,self.config.noisy_copies_length])
    model_input_right = torch.zeros([self.config.max_number_per_cluster,4,self.config.noisy_copies_length])
    
    noisy_copy_length = []
    
    for idx in range(self.config.max_number_per_cluster):
        
        if idx < len(noisy_copies):
            noisy_copy = noisy_copies[idx]
                        
            # Update copies length list
            noisy_copy_length.append(len(noisy_copy))
            
            # Get one-hot embedding
            one_hot_noisy_copy = one_hot_encoding(noisy_copy)
            
            # Get flipped copy
            one_hot_noisy_copy_right = torch.flip(one_hot_noisy_copy,dims=[1])
            
            # Padding
            if one_hot_noisy_copy.shape[-1] < self.config.noisy_copies_length:
                
                model_input[idx,:,:]      = pad_seq(self.config, one_hot_noisy_copy)
                model_input_right[idx,:,:] = pad_seq(self.config, one_hot_noisy_copy_right)

        else:
            noisy_copy_length.append(0)
        
    return model_input, model_input_right, noisy_copy_length, num_noisy_copies

