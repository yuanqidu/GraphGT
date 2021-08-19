import pandas as pd
import numpy as np
import os, sys, json

from . import base_dataset

class DataLoader(base_dataset.DataLoader):
    def __init__(self, name, path, print_stats, dataset_names):
        
        entity1, y, entity1_idx = property_dataset_load(name, path, label_name, dataset_names)
        
        self.entity1 = entity1
        self.y = y
        self.entity1_idx = entity1_idx
        self.name = name
        self.entity1_name = 'Drug'
        self.path = path
        self.file_format = 'csv'
        self.label_name = label_name
        self.convert_format = convert_format
        self.convert_result = None

    def get_data(self, format = 'df'):
        '''
            Arguments:
            df: return pandas DataFrame; if not true, return np.arrays
            returns:
            self.drugs: drug smiles strings np.array
            self.targets: target Amino Acid Sequence np.array
            self.y: inter   action score np.array
            '''
                
        if (self.convert_format is not None) and (self.convert_result is None):
            from ..chem_utils import MolConvert
            converter = MolConvert(src = 'SMILES', dst = self.convert_format)
            convert_result = converter(self.entity1.values)
            self.convert_result = [i for i in convert_result]
                
        if format == 'df':
            return pd.DataFrame({self.entity1_name + '_ID': self.entity1_idx, self.entity1_name: self.entity1, 'Y': self.y})
        else:
            raise AttributeError("Please use the correct format input")

    def print_stats(self):
        print_sys('--- Dataset Statistics ---')
        try:
            x = np.unique(self.entity1)
        except:
            x = np.unique(self.entity1_idx)

        print(str(len(x)) + ' unique ' + self.entity1_name.lower() + 's.', flush = True, file = sys.stderr)
                print_sys('--------------------------')
