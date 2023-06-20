import streamlit as st
import numpy as np
import pandas as pd

from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, ShuffleSplit
from torch import nn
from torch.utils.data import Dataset
from collections import OrderedDict
from model import *
from fastprogress import progress_bar
from utils import lottie_vaccine
# from streamlit_lottie import st_lottie

import functools

# st.set_page_config(layout='wide')

################################################################################

# st_lottie(lottie_vaccine, height=200)


st.title('Stanford covid vaccine')

st.markdown('Disclaimer: This is a project by Low Kheng Oon. This application is not production ready. Use at your own discretion')

st.markdown('Winning the fight against the COVID-19 pandemic will require an effective vaccine that can be equitably and widely distributed. Building upon decades of research has allowed scientists to accelerate the search for a vaccine against COVID-19, but every day that goes by without a vaccine has enormous costs for the world nonetheless. We need new, fresh ideas from all corners of the world. Could online gaming and crowdsourcing help solve a worldwide pandemic? Pairing scientific and crowdsourced intelligence could help computational biochemists make measurable progress.')

st.markdown('mRNA vaccines have taken the lead as the fastest vaccine candidates for COVID-19, but currently, they face key potential limitations. One of the biggest challenges right now is how to design super stable messenger RNA molecules (mRNA). Conventional vaccines (like your seasonal flu shots) are packaged in disposable syringes and shipped under refrigeration around the world, but that is not currently possible for mRNA vaccines.')

st.markdown('Researchers have observed that RNA molecules have the tendency to spontaneously degrade. This is a serious limitation--a single cut can render the mRNA vaccine useless. Currently, little is known on the details of where in the backbone of a given RNA is most prone to being affected. Without this knowledge, current mRNA vaccines against COVID-19 must be prepared and shipped under intense refrigeration, and are unlikely to reach more than a tiny fraction of human beings on the planet unless they can be stabilized.')

st.markdown('The Eterna community, led by Professor Rhiju Das, a computational biochemist at Stanford’s School of Medicine, brings together scientists and gamers to solve puzzles and invent medicine. Eterna is an online video game platform that challenges players to solve scientific problems such as mRNA design through puzzles. The solutions are synthesized and experimentally tested at Stanford by researchers to gain new insights about RNA molecules. The Eterna community has previously unlocked new scientific principles, made new diagnostics against deadly diseases, and engaged the world’s most potent intellectual resources for the betterment of the public. The Eterna community has advanced biotechnology through its contribution in over 20 publications, including advances in RNA biotechnology.')

st.markdown('In this competition, we are looking to leverage the data science expertise of the Kaggle community to develop models and design rules for RNA degradation. Your model will predict likely degradation rates at each base of an RNA molecule, trained on a subset of an Eterna dataset comprising over 3000 RNA molecules (which span a panoply of sequences and structures) and their degradation rates at each position. We will then score your models on a second generation of RNA sequences that have just been devised by Eterna players for COVID-19 mRNA vaccines. These final test sequences are currently being synthesized and experimentally characterized at Stanford University in parallel to your modeling efforts -- Nature will score your models!')

st.markdown('Improving the stability of mRNA vaccines was a problem that was being explored before the pandemic but was expected to take many years to solve. Now, we must solve this deep scientific challenge in months, if not weeks, to accelerate mRNA vaccine research and deliver a refrigerator-stable vaccine against SARS-CoV-2, the virus behind COVID-19. The problem we are trying to solve has eluded academic labs, industry R&D groups, and supercomputers, and so we are turning to you. To help, you can join the team of video game players, scientists, and developers at Eterna to unlock the key in our fight against this devastating pandemic.')

# import data
train_data = pd.read_json('train.json', lines=True)
test_data = pd.read_json('test.json', lines=True)
sub = pd.read_csv('sample_submission.csv')

st.write("Train shapes: ", train_data.shape)
if ~train_data.isnull().values.any(): st.write('No missing values')
st.write(train_data.head(3))

st.write("Test shapes: ", test_data.shape)
if ~test_data.isnull().values.any(): st.write('No missing values')
st.write(test_data.head(3))

st.write("Submission shapes: ", sub.shape)
if ~sub.isnull().values.any(): st.write('No missing values')
st.write(sub.head(3))

st.write(
'''
Now we explore `signal_to_noise` and `SN_filter` distributions. As per the data tab of this project the samples in `test.json` have been filtered in the following way:

1. Minimum value across all 5 conditions must be greater than -0.5.
2. Mean signal/noise across all 5 conditions must be greater than 1.0 [Signal/noise is defined as mean(measurement value over 68 nts)/mean(statistical error in measurement value over 68 nts]
3. To help ensure sequence diversity, the resulting sequences were clustered into clusters with less than 50% sequence similarity, and the 629 test set sequences were chosen from clusters with 3 or fewer members. That is, any sequence in the test set should be sequence similar to at most 2 other sequences.')
''')

fig, ax = plt.subplots(1,2, figsize=(15,5))
sns.kdeplot(x=train_data['signal_to_noise'], shade=True, ax=ax[0])
sns.countplot(x=train_data['SN_filter'], ax=ax[1])

ax[0].set_title('Signal/Noise Distribution')
ax[1].set_title('Signal/Noise Filter Distribution')
st.pyplot(fig)

st.write(f"Samples with signal_to_noise greater than 1: {len(train_data.loc[(train_data['signal_to_noise'] > 1)])}")
st.write(f"Samples with signal_to_noise greater than 1: {len(train_data.loc[(train_data['SN_filter'] == 1)])}")
st.write(f"Samples with signal_to_noise greater than 1, but SN_filter == 0: {len(train_data.loc[(train_data['signal_to_noise'] > 1) & (train_data['SN_filter'] == 0)])}")

st.title('What is BPPs')
st.write('''
In the provided dataset, there's a folder named `bpps`. `bpps` stands for `Base Pairing Probabilities`. 

Below is the explanation from host:
>The `bpps` are numpy arrays pre-calculated for each sequence. They are matrices of base pair probabilities, calculated using a developed algorithm by competition host lab. Biophysically speaking, this matrix gives the probability that each pair of nucleotides in the RNA forms a base pair (given a particular model of RNA folding). The matrix inside `bpps` folder is Base Pairing Probability Metri (BPPM) and it's basically treated as adjascency matrix of the RNA sequence. You've probably already seen the structural features: imagine that this matrix describes the whole distribution from which one could sample more structures. At the simplest level -- it's a symmetric square matrix with the same length as the sequence, so you can get N more features out of it, if you want them. Each column and each row should sum to one (up to rounding error), but more than one entry in each column/row will be nonzero -- usually somewhere between 1-5 entries.

The structure in `json` file and `BPPM` has strong connection. Lets have a look at it.
''')



def predict_batch(model, data, device):
    # batch x seq_len x target_size
    with torch.no_grad():
        pred = model(data["sequence"].to(device), data["bpp"].to(device))
        pred = pred.detach().cpu().numpy()
    return_values = []
    ids = data["ids"]
    for idx, p in enumerate(pred):
        id_ = ids[idx]
        assert p.shape == (model.pred_len, len(target_cols))
        for seqpos, val in enumerate(p):
            assert len(val) == len(target_cols)
            dic = {key: val for key, val in zip(target_cols, val)}
            dic["id_seqpos"] = f"{id_}_{seqpos}"
            return_values.append(dic)
    return return_values


def predict_data(model, loader, device, batch_size):
    data_list = []
    for i, data in enumerate(progress_bar(loader)):
        data_list += predict_batch(model, data, device)
    expected_length = model.pred_len * len(loader) * batch_size
    assert len(data_list) == expected_length, f"len = {len(data_list)} expected = {expected_length}"
    return data_list

input_test = pd.read_json(f'test.json', lines=True)
st.write(input_test.head(1))
input_temp = input_test.iloc[0:1,:].copy()
st.write(input_temp)
use_sample_data = st.checkbox('Use sample data')


if use_sample_data:
    input = input_temp.copy()
    input_seq = input['sequence'][0]
    input_structure = input['structure'][0]
    input_loop_type = input['predicted_loop_type'][0]
    input_seq = st.text_input(label='mRNA sequence to predict', value=f'{input_seq}')
    input_structure = st.text_input(label='mRNA structure for input sequence', value=f'{input_structure}') # http://rna.tbi.univie.ac.at//cgi-bin/RNAWebSuite/RNAfold.cgi
    input_loop_type = st.text_input(label='mRNA loop prediction', value=f'{input_loop_type}')

else:
    input = input_temp.copy()
    input_seq = st.text_input(label='mRNA sequence to predict')
    input_structure = st.text_input(label='mRNA structure for input sequence') # http://rna.tbi.univie.ac.at//cgi-bin/RNAWebSuite/RNAfold.cgi
    input_loop_type = st.text_input(label='mRNA loop prediction')
    input.sequence = input_seq
    input.structure = input_structure
    input.predicted_loop_type = input_loop_type
    input.seq_length = len(input_seq)

@st.cache(allow_output_mutation=True)
def load_model():

    save_dest = Path('model')
    save_dest.mkdir(exist_ok=True)
    
    f_checkpoint = Path("model/model_0.pt")

    if not f_checkpoint.exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            from GD_download import download_file_from_google_drive
            download_file_from_google_drive('1V8EecM-IwkROmMrSMilvBV65remIt7mw', f_checkpoint)
    
    model = torch.load(f_checkpoint, map_location=device)
    # model.eval()
    return model

if ((len(input['sequence'][0]) > 1) and (len(input['sequence'][0]) == len(input['structure'][0])) and (len(input['sequence'][0]) == len(input['predicted_loop_type'][0]))):
    st.write(input[['sequence', 'structure', 'predicted_loop_type', 'seq_length']])

    if st.button('Run Prediction'):
        device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
        BATCH_SIZE = 1
        # base_test_data = pd.read_json(f'models/test.json', lines=True)
        base_test_data = input.copy()
        public_df = base_test_data.query("seq_length == 107").copy()
        # private_df = base_test_data.query("seq_length == 130").copy()
        st.write(f"public_df: {public_df.shape}")
        # st.write(f"private_df: {private_df.shape}")
        public_df = public_df.reset_index()
        # private_df = private_df.reset_index()
        st.write(public_df.head())
        pub_loader = create_loader(public_df, BATCH_SIZE, is_test=True)
        # pri_loader = create_loader(private_df, BATCH_SIZE, is_test=True)
        pred_df_list = []
        c = 0
        # for fold in range(1):
        model_load_path = load_model()
        # model_load_path = f"./models/model-{fold}.pt"
        ae_model0 = AEModel()
        ae_model1 = AEModel()
        model_pub = FromAeModel(pred_len=107, seq=ae_model0.seq)
        model_pub = model_pub.to(device)
        # model_pri = FromAeModel(pred_len=130, seq=ae_model1.seq)
        # model_pri = model_pri.to(device)
        # state_dict = torch.load(model_load_path, map_location=device)
        model_pub.load_state_dict(model_load_path)
        # model_pub.load_state_dict(state_dict)
        # model_pri.load_state_dict(state_dict)
        # del state_dict

        data_list = []
        data_list += predict_data(model_pub, pub_loader, device, BATCH_SIZE)
        # data_list += predict_data(model_pri, pri_loader, device, BATCH_SIZE)
        pred_df = pd.DataFrame(data_list, columns=["id_seqpos"] + target_cols)
        print(pred_df.head())
        print(pred_df.tail())
        pred_df_list.append(pred_df)
        c += 1
        data_dic = dict(id_seqpos=pred_df_list[0]["id_seqpos"])
        for col in target_cols:
            vals = np.zeros(pred_df_list[0][col].shape[0])
            for df in pred_df_list:
                vals += df[col].values
            data_dic[col] = vals / float(c)
        pred_df_avg = pd.DataFrame(data_dic, columns=["id_seqpos"] + target_cols)
        st.write(pred_df_avg)

