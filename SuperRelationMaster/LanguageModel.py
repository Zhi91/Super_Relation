from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch
import torch.nn as nn
import gensim
import sys
import numpy as np


class BERT_Encoder(nn.Module):
    """docstring for BERT_Encoder"""

    def __init__(self,name,num_labels=11, hidden_dropout_prob=0.1):
        super(BERT_Encoder, self).__init__()
        self.name = name
        self.num_labels = num_labels
        if name == 'bert':
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        elif name == 'scibert':
            self.bert = BertModel.from_pretrained('./PretrainedLanguageModel/scibert_scivocab_uncased')
        else:
            print('please add bert_encoder')
            sys.exit()
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        sequence_output, _ = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        return sequence_output

class Word2vec(object):
    """docstring for Gensim"""
    def __init__(self, WORDEMB_PATH, dim = 50):
        super(Word2vec, self).__init__()
        self.fn = WORDEMB_PATH       
        self.dim = dim
        self.word2vec_dict = dict()
        wv_model = gensim.models.KeyedVectors.load_word2vec_format(self.fn, binary=True)
        for word in wv_model.vocab:
            self.word2vec_dict[word] = torch.tensor(wv_model.wv[word], dtype=torch.float32)
        assert len(self.word2vec_dict) == len(wv_model.vocab)
        vector = torch.randn(self.dim, dtype=torch.float32)
        self.word2vec_dict['<pad>'] = vector
        vector = torch.randn(self.dim, dtype=torch.float32)
        assert not '<unk>' in self.word2vec_dict
        self.word2vec_dict['<unk>'] = vector
    def embedding(self,sentence_list:list):
        batch,seq_len = len(sentence_list),len(sentence_list[0])
        embed = torch.zeros([batch,seq_len,self.dim],dtype=torch.float32)
        for i in range(batch):
            for j in range(seq_len):
                if sentence_list[i][j] in self.word2vec_dict:
                    embed[i][j] += self.word2vec_dict[sentence_list[i][j]]
                else:
                    embed[i][j] += self.word2vec_dict['<unk>']
        return embed




class WordEmbedding(object):
    def __init__(self,name:str,device):
        super(WordEmbedding,self).__init__()
        self.name = name
        self.device = device
        print('loading '+name+' ......')
        if name == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
            self.encoder = BERT_Encoder(name)
            self.dim = 768
        elif name == 'scibert':
            self.tokenizer = BertTokenizer.from_pretrained('./PretrainedLanguageModel/scibert_scivocab_uncased',do_lower_case=True)
            self.encoder = BERT_Encoder(name)
            self.dim = 768
        elif name == 'word2vec':
            self.encoder = Word2vec('./PretrainedLanguageModel/Geology_Copus.bin')
            self.dim = self.encoder.dim
        else:
            print('Please add encoder ......')
            sys.exit()
    def Tokenize(self,sentence_list:list):
        seq_len = 0
        word_spans = []
        input_ids = []
        attention_mask = []
        if 'bert' in self.name:
            tokens =[]
            for sen in sentence_list:
                word_piece = []
                for w in sen:
                    word_piece.append(self.tokenizer.tokenize(w))
##record the span of each word###
                sub_len = list(map(len,word_piece))
                word_span = []
                start = 0
                for i in range(len(sub_len)):
                    word_span.append([start,start+sub_len[i]])
                    start = word_span[-1][1]
##record the span of each word###
                sen_token = [token for item in word_piece for token in item]
                assert word_span[-1][1] == len(sen_token)
                assert len(word_span) == len(sen)
                word_spans.append(word_span)
                tokens.append(sen_token)
                seq_len = max(seq_len,word_span[-1][1])
## expand sentence length ##
            for i in range(len(sentence_list)):
                l = len(tokens[i])
                tokens[i] += ['[PAD]']*(seq_len-l)
                attention_mask.append([1]*l+[0]*(seq_len-l))
                input_ids.append(self.tokenizer.convert_tokens_to_ids(tokens[i]))
            input_ids = torch.LongTensor(np.array(input_ids))
        else:
            seq_len = max([len(sen) for sen in sentence_list])
            for sen in sentence_list:
                l = len(sen)
                word_span = []
                for i in range(l):
                    word_span.append([i,i+1])
                word_spans.append(word_span)
## expand sentence length ##
                attention_mask.append([1]*l+[0]*(seq_len-l))
                input_ids.append(sen + ['<pad>']*(seq_len-l))        
        return seq_len,input_ids,word_spans,torch.LongTensor(attention_mask)
    
    def Embedding(self,input_ids,attention_mask):
        if self.name == 'word2vec':
            out_sequence = self.encoder.embedding(input_ids)
        else:    
            out_sequence = self.encoder(input_ids,attention_mask)
        return out_sequence

                


if __name__ == '__main__':
    sens = ['Within an ocean model which is expected to strongly overestimate the wind-driven TCC volume transport , a relatively weak TCC is found for the reconstructed Campanian paleogeography used .',
    'Today we discover how water and magma mixed to produce the largest atmospheric explosion in recorded history .']
    sen_list = [sen.split(' ') for sen in sens]
    device = torch.device('cuda')
    wordemb = WordEmbedding('scibert',device)
    seq_len,input_ids,word_spans = wordemb.Tokenize(sen_list)
    print(seq_len,len(word_spans))
    print(word_spans)
    out_sequence = wordemb.Embedding(input_ids)
    print(out_sequence.size())