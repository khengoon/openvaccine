import numpy as np
import torch
import functools
from collections import OrderedDict
from torch.utils.data import Dataset


target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
input_cols = ['sequence', 'structure', 'predicted_loop_type']
error_cols = ['reactivity_error', 'deg_error_Mg_pH10', 'deg_error_Mg_50C', 'deg_error_pH10', 'deg_error_50C']

token_dicts = {
    "sequence": {x: i for i, x in enumerate("ACGU")},
    "structure": {x: i for i, x in enumerate('().')},
    "predicted_loop_type": {x: i for i, x in enumerate("BEHIMSX")}
}

def preprocess_inputs(df, cols):
    return np.concatenate([preprocess_feature_col(df, col) for col in cols], axis=2)


def preprocess_feature_col(df, col): # col refers to input cols whc are sequence, structure and predicted_loop_type
    dic = token_dicts[col] # numberize col value
    dic_len = len(dic) 
    seq_length = len(df[col][0]) 
    ident = np.identity(dic_len) # get identity array
    # convert to one hot
    arr = np.array(
        df[[col]].applymap(lambda seq: [ident[dic[x]] for x in seq]).values.tolist()
    ).squeeze(1)
    # shape: data_size x seq_length x dic_length
    assert arr.shape == (len(df), seq_length, dic_len)
    return arr


def preprocess(base_data, is_test=False):
    inputs = preprocess_inputs(base_data, input_cols)
    if is_test:
        labels = None
    else:
        labels = np.array(base_data[target_cols].values.tolist()).transpose((0, 2, 1))
        assert labels.shape[2] == len(target_cols)
    assert inputs.shape[2] == 14
    return inputs, labels


def get_bpp_feature(bpp):
    bpp_nb_mean = 0.077522  # mean of bpps_nb across all training data
    bpp_nb_std = 0.08914  # std of bpps_nb across all training data
    bpp_max = bpp.max(-1)[0]
    bpp_sum = bpp.sum(-1)
    bpp_nb = torch.true_divide((bpp > 0).sum(dim=1), bpp.shape[1])
    bpp_nb = torch.true_divide(bpp_nb - bpp_nb_mean, bpp_nb_std)
    return [bpp_max.unsqueeze(2), bpp_sum.unsqueeze(2), bpp_nb.unsqueeze(2)]


@functools.lru_cache(5000)
def load_from_id(id_):
    path = f"bpps/{id_}.npy"
    data = np.load(str(path))
    return data


def get_distance_matrix(leng):
    ## adjacent matrix based on distance on the sequence
    ## D[i, j] = 1 / (abs(i - j) + 1) ** pow, pow = 1, 2, 4

    idx = np.arange(leng)
    Ds = []
    for i in range(len(idx)):
        d = np.abs(idx[i] - idx)
        Ds.append(d)

    Ds = np.array(Ds) + 1
    Ds = 1 / Ds
    Ds = Ds[None, :, :]
    Ds = np.repeat(Ds, 1, axis=0)

    Dss = []
    for i in [1, 2, 4]:
        Dss.append(Ds ** i)
    Ds = np.stack(Dss, axis=3)
    print(Ds.shape)
    return Ds


def get_structure_adj(df):
    ## get adjacent matrix from structure sequence
    ## here I calculate adjacent matrix of each base pair, 
    ## but eventually ignore difference of base pair and integrate into one matrix

    Ss = []
    for i in range(len(df)): # for every row in df
        seq_length = df["seq_length"].iloc[i]
        structure = df["structure"].iloc[i]
        sequence = df["sequence"].iloc[i]
 
        cue = []
        a_structures = OrderedDict([
            (("A", "U"), np.zeros([seq_length, seq_length])),
            (("C", "G"), np.zeros([seq_length, seq_length])),
            (("U", "G"), np.zeros([seq_length, seq_length])),
            (("U", "A"), np.zeros([seq_length, seq_length])),
            (("G", "C"), np.zeros([seq_length, seq_length])),
            (("G", "U"), np.zeros([seq_length, seq_length])),
        ])
        for j in range(seq_length): # for every element in structure
            if structure[j] == "(":
                cue.append(j) # j is the index of pairing foud
            elif structure[j] == ")":
                start = cue.pop() # take the last element in list cue
                a_structures[(sequence[start], sequence[j])][start, j] = 1 # mark pairing pairs as 1
                a_structures[(sequence[j], sequence[start])][j, start] = 1

        a_strc = np.stack([a for a in a_structures.values()], axis=2) # shape: (107,107,6)
        a_strc = np.sum(a_strc, axis=2, keepdims=True) # shape: (107,107,1) # combine into one matrix
        Ss.append(a_strc)

    Ss = np.array(Ss)
    return Ss


def create_loader(df, batch_size=1, is_test=False):
    features, labels = preprocess(df, is_test)
    features_tensor = torch.from_numpy(features)
    if labels is not None:
        labels_tensor = torch.from_numpy(labels)
        dataset = VacDataset(features_tensor, df, labels_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=False)
    else:
        dataset = VacDataset(features_tensor, df, None)
        loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False, drop_last=False)
    return loader


class VacDataset(Dataset):
    def __init__(self, features, df, labels=None):
        self.features = features
        self.labels = labels
        self.test = labels is None
        self.ids = df["id"]
        self.score = None
        self.structure_adj = get_structure_adj(df)
        self.distance_matrix = get_distance_matrix(self.structure_adj.shape[1])
        if "score" in df.columns:
            self.score = df["score"]
        else:
            df["score"] = 1.0
            self.score = df["score"]
        self.signal_to_noise = None
        if not self.test:
            self.signal_to_noise = df["signal_to_noise"]
            assert self.features.shape[0] == self.labels.shape[0]
        else:
            assert self.ids is not None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        bpp = torch.from_numpy(load_from_id(self.ids[index]).copy()).float() # load bpp
        adj = self.structure_adj[index]
        distance = self.distance_matrix[0]
        bpp = np.concatenate([bpp[:, :, None], adj, distance], axis=2)
        if self.test:
            return dict(sequence=self.features[index].float(), bpp=bpp, ids=self.ids[index])
        else:
            return dict(sequence=self.features[index].float(), bpp=bpp,
                        label=self.labels[index], ids=self.ids[index],
                        signal_to_noise=self.signal_to_noise[index],
                        score=self.score[index])
