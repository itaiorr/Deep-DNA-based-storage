from imports_dna import *

class loss_function(nn.Module):  
    def __init__(self,config):
        super(loss_function,self).__init__()
        
        self.config = config

        if self.config.loss_type == 'ce_consistency':
            self.ce_loss = nn.CrossEntropyLoss()
        
            
    def forward(self, model_output, label):
        
        pred       = model_output['pred']
        pred_left  = model_output['pred_left']
        pred_right = model_output['pred_right']
        
        # Cross entropy + consistency
        if self.config.loss_type == 'ce_consistency':
            label_argmax = torch.argmax(label,dim=1).long()
            label = label.float()
            
            ce_loss  = self.config.ce_const_coeff_ce  * self.ce_loss(pred, label_argmax)
            consistency_loss = self.config.ce_const_coeff_const * 0.5 * (self.ce_loss(pred_left.softmax(dim=1), label) + self.ce_loss(pred_right.softmax(dim=1), label))

            # Total loss
            loss = ce_loss + consistency_loss
            
        # Build logger
        if self.config.loss_type=='ce_consistency':
            logger = {'loss':loss, 'ce_loss':ce_loss, 'consistency_loss':consistency_loss}

        return logger
