import pandas as pd
import numpy as np
import sys

class DataLoader:
    def __init__(self):
        pass

    def get_data(self, format='numpy'):
        '''
            Arguments:
            df: return pandas DataFrame; if not true, return np.arrays
            returns:
            self.drugs: drug smiles strings np.array
            self.targets: target Amino Acid Sequence np.array
            self.y: inter   action score np.array
            '''
        if format == 'df':
            return pd.DataFrame({self.entity1_name + '_ID': self.entity1_idx,
                                self.entity1_name: self.entity1, 'Y': self.y})
        else:
            raise AttributeError("Please use the correct format input")

    def print_stats(self):
        print('There are ' + str(len(np.unique(self.entity1))) + ' unique ' + self.entity1_name.lower() + 's',
              flush=True, file=sys.stderr)

    def __len__(self):
        return len(self.get_data(format='df'))
