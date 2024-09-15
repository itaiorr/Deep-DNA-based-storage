
class ErrorRates():
    def __init__(self):
        # initialize general errors
        self.general_errors = {
            'd': '',
            'ld': '',
            'i': '',
            's': ''
        }

        # initialize per-base errors
        self.per_base_errors = {
            'A': {'s': '', 'i': '', 'pi': '', 'd': '', 'ld': ''},
            'C': {'s': '', 'i': '', 'pi': '', 'd': '', 'ld': ''},
            'G': {'s': '', 'i': '', 'pi': '', 'd': '', 'ld': ''},
            'T': {'s': '', 'i': '', 'pi': '', 'd': '', 'ld': ''}
        }

        self.dist_info = {
            'type': '',
            'value': '',
            'min': 0,
            'max': 0
        }
        
    def set_general_errors(self, substitution, insertion, deletion, long_deletion):
        self.general_errors['s'] = substitution
        self.general_errors['i'] = insertion
        self.general_errors['d'] = deletion
        self.general_errors['ld'] = long_deletion
        
    def set_per_base_substitution(self, a, c, g, t):
        self.per_base_errors['A']['s'] = a
        self.per_base_errors['C']['s'] = c
        self.per_base_errors['G']['s'] = g
        self.per_base_errors['T']['s'] = t

    def set_per_base_insertion(self, a, c, g, t):
        self.per_base_errors['A']['i'] = a
        self.per_base_errors['C']['i'] = c
        self.per_base_errors['G']['i'] = g
        self.per_base_errors['T']['i'] = t

    def set_per_base_pre_insertion(self, a, c, g, t):
        self.per_base_errors['A']['pi'] = a
        self.per_base_errors['C']['pi'] = c
        self.per_base_errors['G']['pi'] = g
        self.per_base_errors['T']['pi'] = t

    def set_per_base_del(self, a, c, g, t):
        self.per_base_errors['A']['d'] = a
        self.per_base_errors['C']['d'] = c
        self.per_base_errors['G']['d'] = g
        self.per_base_errors['T']['d'] = t

    def set_per_base_long_del(self, a, c, g, t):
        self.per_base_errors['A']['ld'] = a
        self.per_base_errors['C']['ld'] = c
        self.per_base_errors['G']['ld'] = g
        self.per_base_errors['T']['ld'] = t

    '''MinIONShort + twist_bioscience'''
    def set_B22_values(self):
        # general errors
        self.set_general_errors(1.12e-02, 1.08e-02, 7.87e-03, 2.05e-03)

        # per base errors
        self.set_per_base_substitution(0.01146, 0.01117, 0.01129, 0.01094)
        self.set_per_base_insertion(0.01104, 0.01092, 0.01073, 0.01039)
        self.set_per_base_pre_insertion(0.01096, 0.01088, 0.01071, 0.01053)
        self.set_per_base_del(0.00798, 0.00804, 0.00792, 0.00769)
        self.set_per_base_long_del(0.0021, 0.00208, 0.00201, 0.00201)

    def set_B22_values_partial_data(self):

        # general errors
        self.set_general_errors(1.14e-02, 1.08e-02, 7.67e-03, 2.04e-03)

        # per base errors
        self.set_per_base_substitution(0.01169, 0.01137, 0.01138, 0.01133)
        self.set_per_base_insertion(0.01102, 0.01111, 0.01086, 0.01026)
        self.set_per_base_pre_insertion(0.01119, 0.01166, 0.01, 0.01039)
        self.set_per_base_del(0.00789, 0.00781, 0.00775, 0.00742)
        self.set_per_base_long_del(0.00199, 0.00206, 0.00202, 0.00208)


    ''' Ilumina_miSeq + twist_bioscience '''
    def set_EZ17_values(self):
        # general errors
        self.set_general_errors(1.32e-03, 5.81e-04, 9.58e-04, 2.33e-04)

        # per base errors
        self.set_per_base_substitution(0.00135, 0.00135, 0.00126, 0.00132)
        self.set_per_base_insertion(0.00057, 0.00059, 0.00059, 0.00058)
        self.set_per_base_pre_insertion(0.00059, 0.00058, 0.00057, 0.00058)
        self.set_per_base_del(0.00099, 0.00098, 0.00094, 0.00096)
        self.set_per_base_long_del(0.00024, 0.00023, 0.00023, 0.00023)
        
    def set_EZ17_values_partial_data(self):
        
        # general errors
        self.set_general_errors(1.35e-03, 5.85e-04, 9.62e-04, 2.32e-04)

        # per base errors
        self.set_per_base_substitution(0.00136, 0.00138, 0.0013, 0.00134)
        self.set_per_base_insertion(0.00057, 0.00059, 0.0006, 0.00058)
        self.set_per_base_pre_insertion(0.00059, 0.00059, 0.00058, 0.00058)
        self.set_per_base_del(0.00098, 0.00098, 0.00096, 0.00096)
        self.set_per_base_long_del(0.00023, 0.00023, 0.00023, 0.00023)

    def set_O17_values(self):
        # general errors
        self.set_general_errors(2.52e-03, 4.14e-04, 6.94e-04, 2.11e-04)

        # per base errors
        self.set_per_base_substitution(0.717e-02, 0.034e-02, 0.196e-02, 0.055e-02)
        self.set_per_base_insertion(0.03e-02, 0.007e-02, 0.125e-02, 0.006e-02)
        self.set_per_base_pre_insertion(0.12e-02, 0.007e-02, 0.029e-02, 0.009e-02)
        self.set_per_base_del(0.201e-02, 0.006e-02, 0.058e-02, 0.014e-02)
        self.set_per_base_long_del(0.054e-02, 0.001e-02, 0.023e-02, 0.006e-02)
        
    def set_O17_values_partial_data(self):
        # general errors
        self.set_general_errors(2.52e-03, 4.13e-04, 6.95e-04, 2.11e-04)

        # per base errors
        self.set_per_base_substitution(0.717e-02, 0.034e-02, 0.196e-02, 0.055e-02)
        self.set_per_base_insertion(0.03e-02, 0.007e-02, 0.125e-02, 0.006e-02)
        self.set_per_base_pre_insertion(0.12e-02, 0.007e-02, 0.029e-02, 0.009e-02)
        self.set_per_base_del(0.201e-02, 0.006e-02, 0.058e-02, 0.014e-02)
        self.set_per_base_long_del(0.054e-02, 0.001e-02, 0.024e-02, 0.006e-02)

    ''' Ilumina_miSeq + customArray '''
    def set_G15_values(self):
        # general errors
        self.set_general_errors(5.84e-03, 8.57e-04, 5.37e-03, 3.48e-04)
        
        # per base errors
        self.set_per_base_substitution(0.00605, 0.00563, 0.00577, 0.00591)
        self.set_per_base_insertion(0.0009, 0.00083, 0.00085, 0.00084)
        self.set_per_base_pre_insertion(0.00092, 0.00081, 0.00087, 0.00084)
        self.set_per_base_del(0.00543, 0.00513, 0.00539, 0.00559)
        self.set_per_base_long_del(0.00036, 0.00034, 0.00034, 0.00036)
        
    def set_G15_values_partial_data(self):
        # general errors
        self.set_general_errors(3.66e-03, 7.22e-04, 4.40e-03, 6.75e-04)

        # per base errors
        self.set_per_base_substitution(0.00369, 0.00354, 0.00366, 0.00376)
        self.set_per_base_insertion(0.00073, 0.0007, 0.00073, 0.00073)
        self.set_per_base_pre_insertion(0.00075, 0.00068, 0.00074, 0.00072)
        self.set_per_base_del(0.0048, 0.00463, 0.00484, 0.00507)
        self.set_per_base_long_del(0.00066, 0.00065, 0.00067, 0.00036)
    
    '''MinION + IDT'''
    def set_Y16_values(self):
        # general errors
        self.substitution_doubleSpinBox.setValue(1.21e-01)
        self.insertion_doubleSpinBox.setValue(3.67e-01)
        self.one_base_del_doubleSpinBox.setValue(4.33e-02)
        self.long_del_doubleSpinBox.setValue(1.87e-02)

        # per base errors
        self.set_per_base_substitution(0.119, 0.133, 0.112, 0.119)
        self.set_per_base_insertion(0.331, 0.406, 0.361, 0.367)
        self.set_per_base_pre_insertion(0.332, 0.408, 0.341, 0.382)
        self.set_per_base_del(0.044, 0.048, 0.040, 0.041)
        self.set_per_base_long_del(0.019, 0.021, 0.017, 0.018)

    '''MinION + twist_bioscience'''
    def set_R21_values(self):
        # general errors
        self.set_general_errors(1.08e-02, 1.65e-02, 1.18e-02, 3.36e-03)

        # per base errors
        self.set_per_base_substitution(1.039e-02, 1.042e-02, 1.094e-02, 1.131e-02)
        self.set_per_base_insertion(1.61e-02, 1.639e-02, 1.604e-02, 1.738e-02)
        self.set_per_base_pre_insertion(1.585e-02, 1.578e-02, 1.7e-02, 1.729e-02)
        self.set_per_base_del(1.192e-02, 1.26e-02, 1.267e-02, 1.327e-02)
        self.set_per_base_long_del(0.314e-02, 0.334e-02, 0.337e-02, 0.357e-02)

    def set_R21_values_partial_data(self):
        # general errors
        self.set_general_errors(1.08e-02, 1.65e-02, 1.18e-02, 3.37e-03)

        # per base errors
        self.set_per_base_substitution(1.043e-02, 1.046e-02, 1.102e-02, 1.127e-02)
        self.set_per_base_insertion(1.584e-02, 1.652e-02, 1.59e-02, 1.752e-02)
        self.set_per_base_pre_insertion(1.573e-02, 1.589e-02, 1.72e-02, 1.697e-02)
        self.set_per_base_del(1.181e-02, 1.269e-02, 1.282e-02, 1.315e-02)
        self.set_per_base_long_del(0.313e-02, 0.333e-02, 0.342e-02, 0.359e-02)
        
    def set_R20_values_partial_data(self):

        # general errors
        self.set_general_errors(6.50e-02, 1.04e-01, 6.78e-02, 1.25e-02)

        # per base errors
        self.set_per_base_substitution(0.05955, 0.05867, 0.07254, 0.07215)
        self.set_per_base_insertion(0.0955, 0.09532, 0.1138, 0.11535)
        self.set_per_base_pre_insertion(0.09528, 0.09446, 0.1142, 0.11631)
        self.set_per_base_del(0.064, 0.05924, 0.07674, 0.07618)
        self.set_per_base_long_del(0.01166, 0.01092, 0.01417, 0.01383)
        
        
    '''second pilot (Twist + MiSeq) - 08092022 - Pilot Pool'''
    def set_BOS22_values(self):
        # general errors
        self.set_general_errors(5.29e-04, 5.42e-05, 7.08e-05, 1.81e-05)

        # per base errors
        self.set_per_base_substitution(0.004e-2, 0.024e-2, 0.004e-2, 0.175e-2)
        self.set_per_base_insertion(0.004e-2, 0.006e-2, 0.005e-2, 0.007e-2)
        self.set_per_base_pre_insertion(0.0, 0.002e-2, 0.0, 0.018e-2)
        self.set_per_base_del(0.0, 0.003e-2, 0.0, 0.024e-2)
        self.set_per_base_long_del(0.0, 0.001e-2, 0.0, 0.006e-2)

    def set_BOS22_values_partial_data(self):

        # general errors
        self.set_general_errors(6.00e-04, 5.90e-05, 7.37e-05, 1.96e-05)

        # per base errors
        self.set_per_base_substitution(0.006e-2, 0.03e-2, 0.007e-2, 0.192e-2)
        self.set_per_base_insertion(0.005e-2, 0.007e-2, 0.005e-2, 0.007e-2)
        self.set_per_base_pre_insertion(0.0, 0.003e-2, 0.001e-2, 0.02e-2)
        self.set_per_base_del(0.0, 0.003e-2, 0.0, 0.027e-2)
        self.set_per_base_long_del(0.0, 0.001e-2, 0.0, 0.006e-2)
           
    '''Pilot Pool - Nov2022 (Twist + Nanopore) '''
    def set_BOS22PILOTOMER_values(self):
        # general errors
        self.set_general_errors(1.29e-02, 1.16e-02, 9.75e-03, 2.82e-03)

        # per base errors
        self.set_per_base_substitution(0.52e-2, 1.498e-2, 0.638e-2, 2.437e-2)
        self.set_per_base_insertion(0.644e-2, 1.331e-2, 0.735e-2, 1.867e-2)
        self.set_per_base_pre_insertion(0.582e-2, 1.34e-2, 0.667e-2, 1.983e-2)
        self.set_per_base_del(0.373e-2, 1.178e-2, 0.474e-2, 1.903e-2)
        self.set_per_base_long_del(0.098e-2, 0.336e-2, 0.128e-2, 0.55e-2)
        
    '''Pilot Pool - Nov2022 (Twist + Nanopore) '''
    def set_BOS22PILOTOMER_values_multi(self, noise_coef):
    
        del_mult = noise_coef['del_mult']
        ins_mult = noise_coef['ins_mult']
        sub_mult = noise_coef['sub_mult']
    
        # general errors
        self.set_general_errors(sub_mult*1.29e-02, ins_mult*1.16e-02, del_mult*9.75e-03, del_mult*2.82e-03)

        # per base errors
        self.set_per_base_substitution(sub_mult*0.52e-2, sub_mult*1.498e-2, sub_mult*0.638e-2, sub_mult*2.437e-2)
        self.set_per_base_insertion(ins_mult*0.644e-2, ins_mult*1.331e-2, ins_mult*0.735e-2, ins_mult*1.867e-2)
        self.set_per_base_pre_insertion(ins_mult*0.582e-2, ins_mult*1.34e-2, ins_mult*0.667e-2, ins_mult*1.983e-2)
        self.set_per_base_del(del_mult*0.373e-2, del_mult*1.178e-2, del_mult*0.474e-2, del_mult*1.903e-2)
        self.set_per_base_long_del(del_mult*0.098e-2, del_mult*0.336e-2, del_mult*0.128e-2, del_mult*0.55e-2)
        
        '''Full Pool - Nov2022 (updated) (Twist + Nanopore) '''
    def set_full_dataset_values(self, noise_coef):

        del_mult = noise_coef['del_mult']
        ins_mult = noise_coef['ins_mult']
        sub_mult = noise_coef['sub_mult']

        # general errors
        self.set_general_errors(sub_mult*1.56e-02, ins_mult*1.24e-02, del_mult*9.79e-03, del_mult*2.67e-03)

        # per base errors
        self.set_per_base_substitution(sub_mult*1.7e-2, sub_mult*1.606e-2, sub_mult*1.593e-2, sub_mult*1.358e-2)
        self.set_per_base_insertion(ins_mult*1.32e-2, ins_mult*1.271e-2, ins_mult*1.256e-2, ins_mult*1.116e-2)
        self.set_per_base_pre_insertion(ins_mult*1.343e-2, ins_mult*1.272e-2, ins_mult*1.255e-2, ins_mult*1.094e-2)
        self.set_per_base_del(del_mult*1.095e-2, del_mult*1.038e-2, del_mult*1.014e-2, del_mult*0.849e-2)
        self.set_per_base_long_del(del_mult*0.294e-2, del_mult*0.277e-2, del_mult*0.273e-2, del_mult*0.224e-2)
        
##############################################################################################################

    def set_values(self, sequencing_tech, synthesis_tech, partial_flag, noise_coef):
        if sequencing_tech == 'Ilumina miSeq' and synthesis_tech == 'Twist Bioscience':
            if not partial_flag:
                self.set_EZ17_values()
            else:
                self.set_EZ17_values_partial_data()
                
        elif sequencing_tech == 'Ilumina miSeq' and synthesis_tech == 'CustomArray':
            if not partial_flag:
                self.set_G15_values()
            else:
                self.set_G15_values_partial_data()
                
        elif sequencing_tech == 'Ilumina NextSeq' and synthesis_tech == 'Twist Bioscience':
            if not partial_flag:
                self.set_O17_values()
            else:
                self.set_O17_values_partial_data()
                
        elif sequencing_tech == 'MinION' and synthesis_tech == 'IDT':
            self.set_Y16_values()
            
        elif sequencing_tech == 'MinION' and synthesis_tech == 'Twist Bioscience':
            if not partial_flag:
                self.set_R21_values()
            else:
                self.set_R21_values_partial_data()

        elif sequencing_tech == 'MinIONShort' and synthesis_tech == 'Twist Bioscience':
            if not partial_flag:
                self.set_B22_values()
            else:
                self.set_B22_values_partial_data()
                      
        elif sequencing_tech == 'Ilumina miSeq-0922' and synthesis_tech == 'Twist Bioscience-0922':
            if not partial_flag:
                self.set_BOS22_values()
            else:
                self.set_BOS22_values_partial_data()
                
        elif sequencing_tech == 'Nanopore_pilot_v1' and synthesis_tech == 'Twist Bioscienc_nanopore_pilot_v1':
                if not partial_flag:
                        self.set_BOS22N_values()
                else:
                        self.set_BOS22_values()
            
        elif sequencing_tech == 'Nanopore_full_v1' and synthesis_tech == 'Twist Bioscience_nanopore_full_v1':
                if not partial_flag:
                        self.set_BOS22F_values()
                else:
                        self.set_BOS22F_values()
                               
        elif sequencing_tech == 'Nanopore_pilot_v2' and synthesis_tech == 'Twist Bioscience_nanopore_pilot_v2':
                if not partial_flag:
                        self.set_BOS22PILOTOMER_values()
                else:
                        self.set_BOS22PILOTOMER_values()
                        
        elif sequencing_tech == 'Nanopore_pilot_v2_multi' and synthesis_tech == 'Twist Bioscience_nanopore_pilot_v2_multi':
                if not partial_flag:
                        self.set_BOS22PILOTOMER_values_multi(noise_coef)
                else:
                        self.set_BOS22PILOTOMER_values_multi(noise_coef)
                        
        elif sequencing_tech == 'Nanopore_full_multi' and synthesis_tech == 'Twist Bioscience_nanopore_full_multi':
                if not partial_flag:
                        self.set_full_dataset_values(noise_coef)
                else:
                        self.set_full_dataset_values(noise_coef)
        else:
            raise ValueError('sequencing or synthesis technology are not valid')

