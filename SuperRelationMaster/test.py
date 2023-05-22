import argparse
import os
from unicodedata import name
import torch
from LanguageModel import *
from utils import *
from Model import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime

model_path = r'./models/L2/scibert_mean_dssm_111.pt'
document_file = r'./test.txt'
device = torch.device('cuda' if  torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    data = DataCenter(document_file)
    LM = WordEmbedding('scibert',device)
    LM.encoder.eval()
    seq_len,input_ids,word_span,attention_mask = LM.Tokenize(data.Sentence_List)
    embedding = LM.Embedding(input_ids,attention_mask)
    print(embedding.size())
    model = Multi_Grain('dssm','111',768,200,'L1',2)
    model.load_state_dict(torch.load(model_path))
    # with torch.no_grad():
    for i in range(len(data.Sentence_List)):
        print(' '.join(data.Sentence_List[i]))
        condition= []
        fact = []
        for c in data.Condition_Element[i]:
            c_span = torch.mean(embedding[i,word_span[i][c[0][0]][0]:word_span[i][c[0][1]][1],:],dim=0)
            c_relation = torch.mean(embedding[i,word_span[i][c[3][0]][0]:word_span[i][c[3][1]][1],:],dim=0)
            c_e = []
            for e in [c[1],c[2],c[4],c[5]]:
                if e != []:
                    c_e.append(embedding[i,word_span[i][e[0]][0]:word_span[i][e[1]][1],:])
            c_element = torch.mean(torch.cat(c_e,dim=0),dim=0)
            condition.append(torch.stack([c_span,c_element,c_relation],dim=0))
        fact = []
        for f in data.Fact_Element[i]:
            f_span = torch.mean(embedding[i,word_span[i][f[0][0]][0]:word_span[i][f[0][1]][1],:],dim=0)
            f_relation = torch.mean(embedding[i,word_span[i][f[3][0]][0]:word_span[i][f[3][1]][1],:],dim=0)
            f_e = []
            for e in [f[1],f[2],f[4],f[5]]:
                if e != []:
                    f_e.append(embedding[i,word_span[i][e[0]][0]:word_span[i][e[1]][1],:])
            f_element = torch.mean(torch.cat(f_e,dim=0),dim=0)
            fact.append(torch.stack([f_span,f_element,f_relation],dim=0))
        with torch.no_grad():
            for j in range(len(condition)):
                for k in range(len(fact)):
                    print('Condition:\t')
                    print(' '.join(data.Sentence_List[i][data.Condition_Element[i][j][0][0]:data.Condition_Element[i][j][0][1]])+'\n')
                    print('Fact:\t')
                    print(' '.join(data.Sentence_List[i][data.Fact_Element[i][k][0][0]:data.Fact_Element[i][k][0][1]])+'\n')
                    score = model(torch.cat([fact[k],condition[j]],dim=1).unsqueeze(0))
                    print('Score:%.2f'%score.item())
