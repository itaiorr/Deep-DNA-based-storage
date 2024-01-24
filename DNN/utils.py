from imports import *
import data_loader as data_loader
import loss as loss

###############################################################################################################
def run_train(config, model):

    # Grab loss and accuacry loggers
    config    = utils.grab_loggers(config)
    optimizer = optim.Adam(model.parameters(), lr=config.lrMax, betas=(0.9, 0.999), eps=1e-8)
    
    if config.use_pretrained:
        # Get pretrained chekcpoint
        checkpoint = torch.load(config.pretrained_path)

        # Load the new state dict
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        
        # Load optimizer
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load to GPU
    model = model.cuda(config.gpus_list[0])
    
    # Load optimizer to GPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda(config.gpus_list[0])
    
    for epoch in range(1, config.nEpochs + 1):
    
        # Training
        df_train_epoch        = config.df_train
        train_set             = config.data_loader.DatasetFromFolder(config, df_train_epoch, data_source=config.train_data_source)
        train_data_loader     = DataLoader(dataset=train_set, num_workers=config.nThreads, batch_size=config.batch_size,shuffle=True,worker_init_fn=lambda x: np.random.seed(),pin_memory=True,drop_last=True)
        epoch_train_logger, _ = run_epoch(config, optimizer, loss, epoch, train_data_loader, model, configuration='train')

        # Validation
        df_val_epoch                    = config.df_val
        val_set                         = config.data_loader.DatasetFromFolder(config, df_val_epoch, data_source=config.val_data_source)
        val_data_loader                 = DataLoader(dataset=val_set, num_workers=config.nThreads, batch_size=config.batch_size, shuffle=True,worker_init_fn=lambda x: np.random.seed(),drop_last=True)
        epoch_val_logger, edit_dist_val = run_epoch(config, optimizer, loss, epoch, val_data_loader, model, configuration='val')

        # Learning rate scheduler: cosine annealing (decay if lrCycles=1)
        for param_group in optimizer.param_groups:
                param_group['lr'] = config.lrMin + 0.5*(config.lrMax-config.lrMin)*(1+ np.cos(epoch*np.pi/config.nEpochs*config.lrCycles))
        print('Learning rate decay rd: lr={}'.format(optimizer.param_groups[0]['lr']))

        # Save weights
        config = save_best_model(config, epoch, model, optimizer, epoch_val_logger)

        # Save loss logger       
        for key in epoch_train_logger:
            config.loss_logger['train_'+key].append(epoch_train_logger[key].cpu().numpy().tolist())
            
        for key in epoch_val_logger:
            config.loss_logger['val_'+key].append(epoch_val_logger[key].cpu().numpy().tolist())
        
        with open(config.loss_logger_path, 'w') as outfile:
            json.dump(config.loss_logger, outfile)
            
###############################################################################################################
def run_epoch(config, optimizer, loss, epoch, data_loader, model, configuration):
    
    if configuration=='train':
        model.train()
    elif configuration=='val':
        model.eval()  
        
    epoch_logger = {}
    for key in config.loss_items:
        epoch_logger[key] = torch.zeros(1).cuda(config.gpus_list[0])
                    
    pbar = tqdm(iter(data_loader), leave=True, position=0, total=len(data_loader))
    pbar.set_description('Epoch ' + str(epoch) + ' ' + configuration)
    print('Epoch ' + str(epoch) + ' ' + configuration)

    for iteration,batch in enumerate(pbar, 1):
        
        # Grab batch
        label = batch['label'].cuda(config.gpus_list[0])
                
        # Grab model input
        if config.model_config=='single':
            model_input       = batch['model_input'].cuda(config.gpus_list[0])
        
        elif config.model_config=='siamese':
            model_input       = batch['model_input']
            model_input_right = batch['model_input_right']
            model_input       = torch.cat([model_input, model_input_right],dim=0).cuda(config.gpus_list[0])

        # Run forward
        if configuration=='train':
            optimizer.zero_grad()
            model_output = model(model_input)
            
        elif configuration=='val':
            with torch.inference_mode():
                model_output = model(model_input)
                
        # Get loss
        logger = loss(model_output, label)
            
        # Backward and optimize (train)
        if configuration=='train':
            
            # Backpropagation
            logger['loss'].backward()  

            # Optimization step
            optimizer.step()
            
            # Logging
            stats = {}
            for key in logger:
                epoch_logger[key] += logger[key].detach()
                stats[key] = epoch_logger[key].detach().item()/iteration
                        
        # Evaluate (val)
        elif configuration=='val': 
            
            # Loss
            for key in logger:
                epoch_logger[key] += logger[key].detach()
                
            # Eval
            logger_eval                 = evaluation(model_output['pred'], label)           
            epoch_logger['pred_probs'] += logger_eval['pred_probs'].detach()
                                    
            for i in range(config.batch_size):
                edit_dist.append(logger_eval['edit_dist'][i].item())

            stats = {'loss':epoch_logger['loss'].detach().item()/iteration}

        # Update progress bar
        pbar.set_postfix(ordered_dict=stats, refresh=True)
        
    for key in epoch_logger:
        epoch_logger[key] /= iteration

    return epoch_logger

###############################################################################################################
def run_inference(config, model):

    # Get pretrained chekcpoint
    checkpoint = torch.load(config.pretrained_path)

    # Load the new state dict
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    
    # Load to GPU
    model = model.cuda(config.gpus_list[0])
    
    # Set model to eval
    model.eval()  
        
    # Get dataloader
    df_inf          = config.df_inf.sample(frac=np.min([config.frames_per_epoch['test']/len(config.df_inf),1])).reset_index(drop=True)
    inf_set         = config.data_loader.DatasetFromFolder(config, df_inf, data_source=config.inf_data_source, get_aligned_data=get_aligned_data)  
    inf_data_loader = DataLoader(dataset=inf_set, num_workers=config.nThreads, batch_size=config.batch_size_inf, shuffle=True, worker_init_fn=lambda x: np.random.seed(),drop_last=True)

    # Get progrss bar
    pbar = tqdm(iter(inf_data_loader), leave=False, total=len(inf_data_loader))
    pbar.set_description('Inference')

    # Run over dataset
    for iteration,batch in enumerate(pbar, 1):
        
        # Grab model input
        if config.model_config=='single':
            model_input       = batch['model_input'].cuda(config.gpus_list[0])
        
        elif config.model_config=='siamese':
            model_input       = batch['model_input']
            model_input_right = batch['model_input_right']
            model_input       = torch.cat([model_input, model_input_right],dim=0).cuda(config.gpus_list[0])

        # Run model
        with torch.inference_mode():
            model_output = model(model_input)
            probs        = torch.softmax(model_output['pred'], dim=1)
            probs_left   = torch.softmax(model_output['pred_left'], dim=1)
            probs_right  = torch.softmax(model_output['pred_right'], dim=1)
            
            
        # Do something with results
        
###############################################################################################################
def get_pbar_stats(loss_logger, metric_logger, iteration, batch_size):
    
    stats = {}
    for key in metric_logger:
        stats[key] = metric_logger[key].item()/iteration
    
    for key in loss_logger:
        stats[key] = loss_logger[key].item()/iteration

    return stats

###############################################################################################################
def save_model(config, epoch, model, optimizer):
    torch.save({'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict()},
               config.save_path + config.model_type + "_epoch_{}.pth".format(epoch))
    print("Checkpoint saved to {}".format(config.save_path + config.model_type + "_epoch_{}.pth".format(epoch)))

###############################################################################################################
def get_loss_items(config):
  
    if config.loss_type == 'ce_consistency':
        loss_items = {'loss', 'ce_loss','consistency_loss','pred_probs'}
 
    return loss_items

###############################################################################################################
class edit_distance(nn.Module):
    
    def __init__(self, config):
        super(edit_distance, self).__init__()
        
        self.config = config
    
    def forward(self, reads, labels):
        
        """
        input:
            reads:  (windows, encoding, sequence)
            labels: (windows, encoding, sequence)
        """
        
        # Reset distance matrix
        seq_len = reads.shape[2]+1
        windows = reads.shape[0]
        
        range_vec     = [x for x in range(seq_len)]
        range_samples = [x for x in range(windows)]
        matrix        = torch.zeros((windows, seq_len, seq_len),requires_grad=True).cuda(self.config.gpus_list[0])
        range_mat     = torch.tensor([[range_vec,] * windows]).cuda(self.config.gpus_list[0])
        mask          = torch.zeros(windows).cuda(self.config.gpus_list[0])
        diag          = torch.zeros(windows).cuda(self.config.gpus_list[0])
        val           = torch.zeros(windows).cuda(self.config.gpus_list[0])
        indexes       = torch.zeros(windows).cuda(self.config.gpus_list[0])
        matrix[:,:,0] = range_mat
        matrix[:,0,:] = range_mat
        
        # Get edit distance 
        for x in range_vec[1:]:
            for y in range_vec[1:]:
                mask[:] = torch.as_tensor([not torch.equal(reads[i, :, x-1], labels[i, :, y-1]) for i in range_samples]) 
                diag[:] = torch.add(matrix[:,x-1, y-1], mask.int())
                val[:], indexes[:] = torch.min(torch.stack((torch.add(matrix[:,x-1, y], 1),diag,torch.add(matrix[:,x, y-1], 1)),dim=0),-2)
                matrix[:,x, y] = val.int()
                
        # Get edit distance
        loss = matrix[:,seq_len-1,seq_len-1]
        
        # Normalize by sequence length
        loss = loss / (seq_len-1)

        return loss

###############################################################################################################
def get_windows(config, pred, label):
    
    """
    Split read/label into random windows 
    input:
        pred: (encoding, sequence)
        label: (encoding, sequence)

    output:
        pred_windows:  (windows, encoding, sub_sequence)
        label_windows: (windows, encoding, sub_sequence)
    """

    pred_windows  = torch.zeros((config.loss_window_num, 4, config.loss_window_length), requires_grad=True).cuda(config.gpus_list[0])
    label_windows = torch.zeros((config.loss_window_num, 4, config.loss_window_length), requires_grad=True).cuda(config.gpus_list[0])

    for i in range(config.loss_window_num):

        # Draw random index
        idx_start = torch.randint(0,config.label_length-config.loss_window_length,(1,)).item()

        # Sample windows
        pred_windows[i,:,:]  = pred[:,  idx_start:idx_start + config.loss_window_length]
        label_windows[i,:,:] = label[:, idx_start:idx_start + config.loss_window_length]
        

    return pred_windows, label_windows

###############################################################################################################
class eval_function(nn.Module):  
    def __init__(self,config):
        super(eval_function,self).__init__()
        
        self.config = config
        
        if self.config.eval_type == 'edit':
            self.edit_distance = edit_distance(config)
            
        elif self.config.eval_type == 'edit_cuda': 
            self.blank     = torch.tensor([0], dtype=torch.int).cuda(config.gpus_list[0])
            self.separator = torch.tensor([1], dtype=torch.int).cuda(config.gpus_list[0])
            self.bs        = torch.cat([self.blank, self.separator])
            self.frames_lengths = torch.tensor([config.label_length], dtype=torch.int).cuda(config.gpus_list[0])
            self.labels_lengths = torch.tensor([config.label_length], dtype=torch.int).cuda(config.gpus_list[0])
            self.space     = torch.empty([], dtype=torch.int).cuda(config.gpus_list[0])
                    
    def forward(self, logits, label):
        
        with torch.inference_mode():
            
            if self.config.eval_debug_mode:
                print('logits',logits.shape, logits.dtype, '\tlabel', label.shape, label.dtype)

            # Edit distance cuda implementation: https://github.com/1ytic/pytorch-edit-distance
            pred_probs = torch.softmax(logits,dim=1)
            pred  = torch.argmax(pred_probs,dim=1) + 1 
            label = torch.argmax(label,dim=1) + 1
            
            # Get number of wrong predictions
            w_cluster = ((pred==label).sum(dim=1)!=self.config.label_length).sum()

            criterion = 0
            edit_dist = []
            if False:
              # Get edit distance
              for i in range(self.config.batch_size):
                  edit_distance = levenshtein_distance(pred[i,...].unsqueeze(0).int(),
                                                       label[i,...].unsqueeze(0).int(), 
                                                       self.frames_lengths, self.labels_lengths, self.bs, self.space).float()
  
                  # Edit distance is sum over sub/del/ins
                  edit_distance = torch.sum(edit_distance[:,:3],dim=1)
  
                  # Normalize to sequence length
                  if self.config.norm_eval_seq_length:
                      criterion += edit_distance.float()/self.config.label_length
                  else:
                      criterion += edit_distance.float()
                      
                  # Edit logger
                  edit_dist.append(edit_distance)
  
            # Hamming approximation
            else:
                  edit_distance = F.l1_loss(pred.float(),label.float(), reduction='none').sum(dim=1)
                  criterion = edit_distance.mean()
                  
                  for i in range(self.config.batch_size):
  
                    # Edit logger
                    edit_dist.append(edit_distance[i])

            # Average to batch size
            criterion /= self.config.batch_size
                        
            # Log prediction confidence
            pred_probs = pred_probs.mean(dim=0).max(dim=0).values
                

            # Build logger
            logger = {'criterion':criterion, 
                      'edit_dist':edit_dist,
                      'pred_probs':pred_probs,
                      'w_cluster':w_cluster}
                
        return logger
       
