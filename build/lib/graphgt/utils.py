import os, sys
import requests
import numpy as np
from .meta import *
from tqdm import tqdm


def print_sys(s):
    print(s, flush=True, file=sys.stderr)

def load_single_dataset(name, save_path='./'):
    download_path = single_dataset_download_path[name]
    save_path = os.path.join(save_path, name)
    adj, node_feat, edge_feat, spatial, label = None, None, None, None, None
    if os.path.exists(save_path):
        print_sys('Found local copy...')
    else:
        os.mkdir(save_path)
        if 'node_feat' in download_path:
            print_sys('Downloading node feature...')
            dataverse_download(download_path['node_feat'], os.path.join(save_path,'node_feat.npy'))
            print_sys("Done!")
        if 'edge_feat' in download_path:
            print_sys('Downloading edge feature...')
            dataverse_download(download_path['edge_feat'], os.path.join(save_path,'edge_feat.npy'))
            print_sys("Done!")
        if 'spatial' in download_path:
            print_sys('Downloading spatial feature...')
            dataverse_download(download_path['spatial'], os.path.join(save_path,'spatial.npy'))
            print_sys("Done!")
        if 'adj' in download_path:
            print_sys('Downloading adjacency matrix...')
            dataverse_download(download_path['adj'], os.path.join(save_path,'adj.npy'))
            print_sys("Done!")
        if 'label' in download_path:
            print_sys('Downloading smiles string...')
            dataverse_download(download_path['label'], os.path.join(save_path,'label.npy'))
            print_sys("Done!")

    if 'adj' in download_path:
        adj = np.load(os.path.join(save_path,'adj.npy'),allow_pickle=True)
    if 'node_feat' in download_path:
        node_feat = np.load(os.path.join(save_path,'node_feat.npy'),allow_pickle=True)
    if 'edge_feat' in download_path:
        edge_feat = np.load(os.path.join(save_path,'edge_feat.npy'),allow_pickle=True)
    if 'spatial' in download_path:
        spatial = np.load(os.path.join(save_path,'spatial.npy'),allow_pickle=True)
    if 'label' in download_path:
        label = np.load(os.path.join(save_path,'label.npy'),allow_pickle=True)
    return adj, node_feat, edge_feat, spatial, label

def load_pair_dataset(name, save_path='./'):
    download_path = pair_dataset_download_path[name]
    save_path = os.path.join(save_path, name)
    input_adj, input_node_feat, input_edge_feat, input_spatial, label = None, None, None, None, None
    target_adj, target_node_feat, target_edge_feat, target_spatial = None, None, None, None
    if os.path.exists(save_path):
        print_sys('Found local copy...')
    else:
        os.mkdir(save_path)
        if 'input_node_feat' in download_path:
            print_sys('Downloading input node feature...')
            dataverse_download(download_path['input_node_feat'], os.path.join(save_path,'input_node_feat.npy'))
            print_sys("Done!")
        if 'input_edge_feat' in download_path:
            print_sys('Downloading input edge feature...')
            dataverse_download(download_path['input_edge_feat'], os.path.join(save_path,'input_edge_feat.npy'))
            print_sys("Done!")
        if 'input_spatial' in download_path:
            print_sys('Downloading input spatial...')
            dataverse_download(download_path['input_spatial'], os.path.join(save_path,'input_spatial.npy'))
            print_sys("Done!")
        if 'input_adj' in download_path:
            print_sys('Downloading input adjacency matrix...')
            dataverse_download(download_path['input_adj'], os.path.join(save_path,'input_adj.npy'))
            print_sys("Done!")
        if 'target_node_feat' in download_path:
            print_sys('Downloading target node feature...')
            dataverse_download(download_path['target_node_feat'], os.path.join(save_path,'target_node_feat.npy'))
            print_sys("Done!")
        if 'target_edge_feat' in download_path:
            print_sys('Downloading target edge feature...')
            dataverse_download(download_path['target_edge_feat'], os.path.join(save_path,'target_edge_feat.npy'))
            print_sys("Done!")
        if 'target_spatial' in download_path:
            print_sys('Downloading target spatial feature...')
            dataverse_download(download_path['target_spatial'], os.path.join(save_path,'target_spatial.npy'))
            print_sys("Done!")
        if 'target_adj' in download_path:
            print_sys('Downloading adjacency matrix...')
            dataverse_download(download_path['target_adj'], os.path.join(save_path,'target_adj.npy'))
            print_sys("Done!")
        if 'label' in download_path:
            print_sys('Downloading smiles string...')
            dataverse_download(download_path['label'], os.path.join(save_path,'label.npy'))
            print_sys("Done!")

    if 'input_adj' in download_path:
        input_adj = np.load(os.path.join(save_path,'input_adj.npy'),allow_pickle=True)
    if 'input_node_feat' in download_path:
        input_node_feat = np.load(os.path.join(save_path,'input_node_feat.npy'),allow_pickle=True)
    if 'input_edge_feat' in download_path:
        input_edge_feat = np.load(os.path.join(save_path,'input_edge_feat.npy'),allow_pickle=True)
    if 'input_spatial' in download_path:
        input_spatial = np.load(os.path.join(save_path,'input_spatial.npy'),allow_pickle=True)
    if 'target_adj' in download_path:
        target_adj = np.load(os.path.join(save_path,'target_adj.npy'),allow_pickle=True)
    if 'target_node_feat' in download_path:
        target_node_feat = np.load(os.path.join(save_path,'target_node_feat.npy'),allow_pickle=True)
    if 'target_edge_feat' in download_path:
        target_edge_feat = np.load(os.path.join(save_path,'target_edge_feat.npy'),allow_pickle=True)
    if 'target_spatial' in download_path:
        target_spatial = np.load(os.path.join(save_path,'target_spatial.npy'),allow_pickle=True)
    if 'label' in download_path:
        label = np.load(os.path.join(save_path,'label.npy'),allow_pickle=True)
    return input_adj, input_node_feat, input_edge_feat, input_spatial, target_adj, target_node_feat, target_edge_feat, target_spatial, label

def dataverse_download(url, save_path):
    
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()


