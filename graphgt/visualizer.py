import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

def plot_single_dist(x):
    sns.distplot(x)
    plt.show()

def plot_overlap_dist(x, y):
    sns.distplot(x)
    sns.distplot(y)
    plt.show()

def visualize_mol(path, smile):
    mol = Chem.MolFromSmiles(smile)
    AllChem.Compute2DCoords(mol)
    Draw.MolToFile(mol,path)


# tester
#batch = 1000
#x = np.random.rand(batch,1)
#plot_single_dist(x)
#y_baseline = np.random.rand(batch,1)
#plot_overlap_dist(x, y_baseline)

