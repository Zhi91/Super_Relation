import torch
import torch.nn as nn
import sys



class DSSM(nn.Module):
    def __init__(self,name,word_dim,hidden_dim):
        super(DSSM,self).__init__()
        self.hidden_dim = hidden_dim
        self.name = name
        self.fact_layer = nn.Sequential(nn.Linear(word_dim,hidden_dim),nn.Dropout(0.4))
        self.condition_layer = nn.Sequential(nn.Linear(word_dim,hidden_dim),nn.Dropout(0.4))
    def forward(self,fact_emb:torch.Tensor,condition_emb:torch.Tensor):
        return self.fact_layer(fact_emb),self.condition_layer(condition_emb)

class Nonlinear_DSSM(nn.Module):
    def __init__(self,name,word_dim,hidden_dim):
        super(Nonlinear_DSSM,self).__init__()
        self.name = name
        self.hidden_dim = hidden_dim
        self.word_dim = word_dim
        self.fact_layer = nn.Sequential(nn.Linear(word_dim,hidden_dim),nn.Dropout(0.4),nn.Tanh())
        self.condition_layer = nn.Sequential(nn.Linear(word_dim,hidden_dim),nn.Dropout(0.4),nn.Tanh())
    def forward(self,fact_emb:torch.Tensor,condition_emb:torch.Tensor):
        return self.fact_layer(fact_emb),self.condition_layer(condition_emb)

class SVM(nn.Module):
    def __init__(self,name,word_dim,target_size):
        super(SVM,self).__init__()
        self.name = name
        self.svm_layer = nn.Sequential(nn.Linear(word_dim,target_size),nn.Dropout(0.4))
    def forward(self,word_emb):
        return self.svm_layer(word_emb)


class Multi_Grain(nn.Module):
    def __init__(self,model_name,multi_grain,word_dim,hidden_dim,dist,target_dim=2):
        super(Multi_Grain,self).__init__()
        self.gate =[int(g) for g in multi_grain]
        self.gate_num = sum(self.gate)
        self.name = model_name+'_'+multi_grain
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.dist = dist
        if model_name == 'dssm':
            if self.gate[0]:
                self.Span = DSSM('span',word_dim,hidden_dim)
            else: self.Span = None
            if self.gate[1]:
                self.Element = DSSM('element',word_dim,hidden_dim)
            else: self.Element = None
            if self.gate[2]:
                self.Relation = DSSM('relation',word_dim,hidden_dim)
            else: self.Relation = None
        elif model_name == 'svm':
            if self.gate[0]:
                self.Span = SVM('span',word_dim,hidden_dim)
            else: self.Span = None
            if self.gate[1]:
                self.Element = SVM('element',word_dim,hidden_dim)
            else: self.Element = None
            if self.gate[2]:
                self.Relation = SVM('relation',word_dim,hidden_dim)
            else: self.Relation = None
            self.multi_head = SVM('multi_head',hidden_dim,target_dim)
        elif model_name == 'nl_dssm':
            if self.gate[0]:
                self.Span = Nonlinear_DSSM('span',word_dim,hidden_dim)
            else: self.Span = None
            if self.gate[1]:
                self.Element = Nonlinear_DSSM('element',word_dim,hidden_dim)
            else: self.Element = None
            if self.gate[2]:
                self.Relation = Nonlinear_DSSM('relation',word_dim,hidden_dim)
            else: self.Relation = None
        elif model_name == 'dssm_svm':
            if self.gate[0]:
                self.Span = DSSM('span',word_dim,hidden_dim)
            else: self.Span = None
            if self.gate[1]:
                self.Element = DSSM('element',word_dim,hidden_dim)
            else: self.Element = None
            if self.gate[2]:
                self.Relation = DSSM('relation',word_dim,hidden_dim)
            else: self.Relation = None
            self.multi_head = SVM('multi_head',hidden_dim,target_dim)
        elif model_name == 'nl_dssm_svm':
            if self.gate[0]:
                self.Span = Nonlinear_DSSM('span',word_dim,hidden_dim)
            else: self.Span = None
            if self.gate[1]:
                self.Element = Nonlinear_DSSM('element',word_dim,hidden_dim)
            else: self.Element = None
            if self.gate[2]:
                self.Relation = Nonlinear_DSSM('relation',word_dim,hidden_dim)
            else: self.Relation = None
            self.multi_head = SVM('multi_head',hidden_dim,target_dim)            
        else:
            print('please add model')
            sys.exit()

    def forward(self,word_emb):
        if 'dssm' in self.name:
            fact,condition,d = [],[],[]
            if self.Span:
                f_span,c_span = self.Span(word_emb[:,0,:self.word_dim],word_emb[:,0,self.word_dim:])
                fact.append(f_span)
                condition.append(c_span)
                if self.dist == 'L1':
                    d.append(torch.abs(f_span-c_span))
                elif self.dist == 'L2':
                    d.append((f_span-c_span)**2)
                else:
                    print('Please dist type')
                    sys.exit()
            else:
                pass
            if self.Element:
                f_element,c_element = self.Element(word_emb[:,1,:self.word_dim],word_emb[:,1,self.word_dim:])
                fact.append(f_element)
                condition.append(c_element)
                if self.dist == 'L1':
                    d.append(torch.abs(f_element-c_element))
                elif self.dist == 'L2':
                    d.append((f_element-c_element)**2)
                else:
                    print('Please dist type')
                    sys.exit()
            else:pass
            if self.Relation:
                f_relation,c_relation = self.Relation(word_emb[:,2,:self.word_dim],word_emb[:,2,self.word_dim:])
                fact.append(f_relation)
                condition.append(c_relation)
                if self.dist == 'L1':
                    d.append(torch.abs(f_relation-c_relation))
                elif self.dist == 'L2':
                    d.append((f_relation-c_relation)**2)
                else:
                    print('Please dist type')
                    sys.exit()
            else:pass

            if self.gate_num > 1 :
                if 'svm' not in self.name:
                    return torch.sum(torch.cosine_similarity(torch.stack(fact,dim=1),torch.stack(condition,dim=1),dim=-1),dim=-1)
                else:
                    return self.multi_head(torch.sum(torch.stack(d,dim=0),dim=0))
            else:
                if 'svm' not in self.name:
                    return torch.cosine_similarity(fact[0],condition[0],dim=-1)
                else:
                    return self.multi_head(d[0])
        else:
            hidden_emb = []
            if self.Span:
                span = self.Span(word_emb[:,0,])
                hidden_emb.append(span)
            else:pass
            if self.Element:
                element = self.Element(word_emb[:,1,])
                hidden_emb.append(element)
            else:pass
            if self.Relation:
                relation = self.Relation(word_emb[:,2,])
                hidden_emb.append(relation)            
            return self.multi_head(torch.sum(torch.stack(hidden_emb,dim=0),dim=0))

class Cosine_loss(nn.Module):
    def __init__(self,name:str):
        super(Cosine_loss,self).__init__()
        self.name = name
    def forward(self,pred:torch.Tensor,truth:torch.Tensor):
        return torch.mean(torch.abs(pred-truth),0)  