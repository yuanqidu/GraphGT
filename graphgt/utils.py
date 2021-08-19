import requests
from zipfile import ZipFile
import os, sys
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from tqdm import tqdm
from meta import dataset_names, node_feat_dataset_names, edge_feat_dataset_names, label_dataset_names, spatial_dataset_names

def fuzzy_search(name, dataset_names):
    name = name.lower()
#    if name[:4] == 'tdc.':
#        name = name[4:]
    if name in dataset_names:
        s = name
    else:
        # print("========fuzzysearch=======", dataset_names, name)
        s = get_closet_match(dataset_names, name)[0]
    if s in dataset_names:
        return s
    else:
        raise ValueError(s + " does not belong to this task, please refer to the correct task name!")

def zip_data_download_wrapper(name, path, server_path, dataset_names):
    name = fuzzy_search(name, dataset_names)
    print (name)
    
    dataset_path = server_path
    
    if not os.path.exists(path):
        os.mkdir(path)
    
    if os.path.exists(os.path.join(path, name)):
        print_sys('Found local copy...')
    else:
        print_sys('Downloading...')
        dataverse_download(dataset_path, path, name)
        print_sys('Extracting zip file...')
        print (os.path.join(path, name + '.zip'))
        with ZipFile(os.path.join(path, name + '.zip'), 'r') as zip:
            zip.extractall(path = os.path.join(path))
        print_sys("Done!")
    return name

def get_closet_match(predefined_tokens, test_token, threshold=0.8):
    """Get the closest match by Levenshtein Distance.
        Parameters
        ----------
        predefined_tokens : list of string
        Predefined string tokens.
        test_token : string
        User input that needs matching to existing tokens.
        threshold : float in (0, 1), optional (default=0.8)
        The lowest match score to raise errors.
        Returns
        -------
        """
    prob_list = []
    
    for token in predefined_tokens:
        # print(token)
        prob_list.append(
                         fuzz.ratio(str(token).lower(), str(test_token).lower()))
    
    assert (len(prob_list) == len(predefined_tokens))

    prob_max = np.nanmax(prob_list)
    print (predefined_tokens)
    token_max = predefined_tokens[np.nanargmax(prob_list)]

    # match similarity is low
    if prob_max / 100 < threshold:
        print_sys(predefined_tokens)
        raise ValueError(test_token,
                         "does not match to available values. "
                         "Please double check.")
    return token_max, prob_max / 100

def dataverse_download(url, path, name):
    save_path = os.path.join(path, name + '.zip')
    print (url)
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

def load_single_data(name, path):
    node_feat, adj, edge_feat, spatial, label = None, None, None, None, None
    if name in node_feat_dataset_names:
        node_feat = np.load(os.path.join(os.path.join(path,name),'node_feat.npy'))
    if name in edge_feat_dataset_names:
        edge_feat = np.load(os.path.join(os.path.join(path,name),'edge_feat.npy'))
    if name in spatial_dataset_names:
        spatial = np.load(os.path.join(os.path.join(path,name),'spatial.npy'))
    if name in label_dataset_names:
        label = np.load(os.path.join(os.path.join(path,name),'label.npy'))
    adj = np.load(os.path.join(os.path.join(path,name),'adj.npy'))
    return node_feat, adj, edge_feat, spatial, label

def load_multi_data(name, path):
    input_node_feat, input_adj, input_edge_feat, input_spatial, input_label = None, None, None, None, None
    target_node_feat, target_adj, target_edge_feat, target_spatial, target_label = None, None, None, None, None
    if name in node_feat_dataset_names:
        input_node_feat = np.load(os.path.join(os.path.join(path,name),'input_node_feat.npy'))
        target_node_feat = np.load(os.path.join(os.path.join(path,name),'target_node_feat.npy'))
    if name in edge_feat_dataset_names:
        input_edge_feat = np.load(os.path.join(os.path.join(path,name),'input_edge_feat.npy'))
        target_edge_feat = np.load(os.path.join(os.path.join(path,name),'target_edge_feat.npy'))
    if name in spatial_dataset_names:
        input_spatial = np.load(os.path.join(os.path.join(path,name),'input_spatial.npy'))
        target_spatial = np.load(os.path.join(os.path.join(path,name),'target_spatial.npy'))
    if name in label_dataset_names:
        input_label = np.load(os.path.join(os.path.join(path,name),'input_label.npy'))
        target_label = np.load(os.path.join(os.path.join(path,name),'target_label.npy'))
    input_adj = np.load(os.path.join(os.path.join(path,name),'input_adj.npy'))
    target_adj = np.load(os.path.join(os.path.join(path,name),'target_adj.npy'))
    return input_node_feat, input_adj, input_edge_feat, input_spatial, input_label, target_node_feat, target_adj, target_edge_feat, target_spatial, target_label


def single_dataset_load(name, path, server_path, dataset_names):
    name = zip_data_download_wrapper(name, path, server_path, dataset_names)
    print_sys('Loading...')
    node_feat, edge_feat, edge_adj, spatial_coord, label = load_data(name, path)
    return node_feat, edge_feat, edge_adj, spatial_coord, label

def multi_dataset_load(name, path, server_path, dataset_names):
    name = zip_data_download_wrapper(name, path, server_path, dataset_names)
    print_sys('Loading...')
    return load_multi_data(name, path)

def print_sys(s):
    print(s, flush=True, file=sys.stderr)

# multi-load
a = multi_dataset_load('BA_small', './data', 'https://drive.google.com/u/0/uc?id=1sAz8oSSD4rkFJyOyG7CVdC6KZ5OThlkp&export=download', dataset_names['synthetic'])
