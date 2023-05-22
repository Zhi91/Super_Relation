import argparse
from ast import arg
import os
from unicodedata import name
import torch
from LanguageModel import *
from utils import *
from Model import *
import torch.optim as optim
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

hidden_dim = 200
target_size = 2
batch_size = 50
lr = 0.005
result_folder = r'./result'
model_folder = r'./models'
pred_folder = r'./pred'

parser = argparse.ArgumentParser(description =
                    'Implement of SVM and DSSM for Super Relation Extraction')

parser.add_argument('--data', type=str, default=r'./Data/data.tsv',
                    help='location of the labeled data')
parser.add_argument('--division', type=str, default=r'./Data/index.tsv',
                    help='division of trainset testtset validset')
parser.add_argument('--epoch', type=int, default=1000,
                    help='number of the train epoch')
parser.add_argument('--pooling', type=str, default='mean',
                    help='pooling of element cluster,soft_m,mean,max')
parser.add_argument('--encoder', type=str, default='scibert',
                    help='type of pretrained Language Model')
parser.add_argument('--model', type=str, default='dssm',
                    help='type of Model')
parser.add_argument('--gate', type=str, default='111',
                    help='type of multi-grain representation')
parser.add_argument('--dist', type=str, default='L2',
                    help='type of multi-grain representation')

args = parser.parse_args()


if __name__ == '__main__':
    gate_num = sum([int(i) for i in args.gate])
    Data = DataCenter(args.data)
    division_id = Sampling(args.division)
    print(division_id.keys())
    TrainSet = Dataset(Data,'train_set',division_id['train_idx:'])
    ValidSet = Dataset(Data,'valid_set',division_id['valid_idx:'])
    TestSet = Dataset(Data,'test_set',division_id['test_idx:'])
    print(len(TrainSet.Sentence_List),len(ValidSet.Sentence_List),len(TestSet.Sentence_List))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    LM = WordEmbedding(args.encoder,device)
    LM.encoder.eval()

    '''word embedding'''
    seq_len,input_ids, word_span,attention_mask = LM.Tokenize(TrainSet.Sentence_List)
    train_embedding = LM.Embedding(input_ids,attention_mask)
    train_embedding = train_embedding.detach()
    print(train_embedding.size())
    
    t_seq_len,t_input_ids, t_word_span,t_attention_mask = LM.Tokenize(TestSet.Sentence_List)
    t_embedding = LM.Embedding(t_input_ids,t_attention_mask )
    t_embedding = t_embedding.detach()
    print(t_embedding.size())

    v_seq_len,v_input_ids, v_word_span,v_attention_mask  = LM.Tokenize(ValidSet.Sentence_List)
    v_embedding = LM.Embedding(v_input_ids,v_attention_mask)
    v_embedding = v_embedding.detach()
    print(v_embedding.size())

    '''fact and condition element extractation'''
    train_emb,train_sen,train_fc = Embedding_Extraction(TrainSet,word_span,train_embedding,args.pooling,device)
    test_emb,test_sen,test_fc = Embedding_Extraction(TestSet,t_word_span,t_embedding,args.pooling,device)
    valid_emb,valid_sen,valid_fc = Embedding_Extraction(ValidSet,v_word_span,v_embedding,args.pooling,device)
    
    print('train \t test \t valid')
    print(train_emb.size(),test_emb.size(),valid_emb.size())
    if args.model == 'svm':
        if args.dist == 'L1':
            train_emb = (torch.abs(train_emb[:,:,:LM.dim] - train_emb[:,:,LM.dim:])).to(device)
            valid_emb = (torch.abs(valid_emb[:,:,:LM.dim] - valid_emb[:,:,LM.dim:])).to(device)
            test_emb = (torch.abs(test_emb[:,:,:LM.dim] - test_emb[:,:,LM.dim:])).to(device)
        elif args.dist == 'L2':
            train_emb = ((train_emb[:,:,:LM.dim] - train_emb[:,:,LM.dim:])**2).to(device)
            valid_emb = ((valid_emb[:,:,:LM.dim] - valid_emb[:,:,LM.dim:])**2).to(device)
            test_emb = ((test_emb[:,:,:LM.dim] - test_emb[:,:,LM.dim:])**2).to(device)
        elif args.dist == 'concat': 
            train_emb = train_emb.to(device)
            valid_emb = valid_emb.to(device)
            test_emb = test_emb.to(device)
        else:
            print('please add dist for svm')
            sys.exit()            

    '''super reltaion extractation, training'''
    torch.cuda.manual_seed(seed)
    if  args.dist != 'concat':
        model = Multi_Grain(args.model,args.gate,LM.dim,hidden_dim,args.dist,target_size).to(device)
    else:
        model = Multi_Grain(args.model,args.gate,LM.dim*2,hidden_dim,args.dist,target_size).to(device)
    optimizer = optim.SGD(model.parameters(),lr,weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss() if 'svm' in args.model else Cosine_loss(args.model)
    batch_num = len(TrainSet.SR_List)//batch_size
    train_truth = torch.LongTensor(np.array(TrainSet.SR_List)).to(device) if 'svm' in args.model else torch.LongTensor(np.array([t if t else -1 for t in TrainSet.SR_List])).to(device)
    print(train_truth.size())
    best_valid_a,best_valid_p,best_valid_r,best_valid_F1 = 0,0,0,0
    best_test_a,best_test_p,best_test_r,best_test_F1 = 0,0,0,0
    best_valid_pred,best_test_pred,loss_list,train_accuracy,valid_accuracy,test_accuracy,train_F1,valid_F1,test_F1 = [],[],[],[],[],[],[],[],[]
    
    result_file = os.path.join(result_folder,args.encoder+'_'+args.pooling+'_'+args.model+'_'+args.gate+'_'+str(batch_size)+'_'+str(hidden_dim)+'_'+datetime.datetime.now().strftime('%Y%m%d%H%M')+'.tsv')
    pred_file = os.path.join(pred_folder,args.encoder+'_'+args.pooling+'_'+args.model+'_'+args.gate+'_'+str(batch_size)+'_'+str(hidden_dim)+'_'+datetime.datetime.now().strftime('%Y%m%d%H%M')+'.txt')
    model_path = os.path.join(model_folder,args.encoder+'_'+args.pooling+'_'+model.name+'_'+str(batch_size)+'_'+str(hidden_dim)+'_'+datetime.datetime.now().strftime('%Y%m%d%H%M')+'.pt')


    for i in range(args.epoch):
        model.train()
        total_loss = 0.
        print('===================')
        print('epoch:%03d'%(i+1))
        for j in range(batch_num):
            pred = model(train_emb[j*batch_size:(j+1)*batch_size,:,:])
            optimizer.zero_grad()
            loss = criterion(pred,train_truth[j*batch_size:(j+1)*batch_size]).to(device)
            print('batch:%03d,loss:%.2f'%(j+1,loss.item()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()
        pred = model(train_emb[batch_num*batch_size:,:,:])
        optimizer.zero_grad()
        loss = criterion(pred,train_truth[batch_num*batch_size:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        loss_list.append(total_loss)
        print('epoch:%03d  total loss:%.2f'%(i+1,total_loss))
        '''evalutate the trained model for each epoch'''
        model.eval()
        with torch.no_grad():
            if 'svm' not in model.name :
                pred_t = model(train_emb)
                pred_v = model(valid_emb)
                pred_test = model(test_emb)
            elif 'svm' in model.name:
                pred_t = torch.max(torch.softmax(model(train_emb),dim=-1),dim=-1)[1]
                pred_v = torch.max(torch.softmax(model(valid_emb),dim=-1),dim=-1)[1]
                pred_test = torch.max(torch.softmax(model(test_emb),dim=-1),dim =-1)[1]
            else:
                print('please add model')
                sys.exit()
        '''move data to cpu for evaluation'''
        train_pred = []
        valid_pred = []
        test_pred = []
        shredhold,V_shredhold,T_shredhold = 0,0,0
        for k in range(pred_t.size()[0]):
            train_pred.append(pred_t[k].item())
        for k in range(pred_v.size()[0]):
            valid_pred.append(pred_v[k].item())
        for k in range(pred_test.size()[0]):
            test_pred.append(pred_test[k].item())

        if 'svm' not in model.name:
            shredhold,a,p,r,F1 = Find_shredhold(train_pred,TrainSet.SR_List)
            v_a,v_p,v_r,v_F1 = Evaluatation(valid_pred,ValidSet.SR_List,shredhold)
            t_a,t_p,t_r,t_F1 = Evaluatation(test_pred,TestSet.SR_List,shredhold)
        elif 'svm' in model.name:
            a,p,r,F1 = Evaluatation(train_pred,TrainSet.SR_List)
            v_a,v_p,v_r,v_F1 = Evaluatation(valid_pred,ValidSet.SR_List)
            t_a,t_p,t_r,t_F1 = Evaluatation(test_pred,TestSet.SR_List)
        else:
            print('please add model')
            sys.exit()
        train_accuracy.append(a)
        valid_accuracy.append(v_a)
        test_accuracy.append(t_a)
        train_F1.append(F1)
        valid_F1.append(v_F1)
        test_F1.append(t_F1)

        print('epoch:%03d,total_loss:%.2f'%(i+1,total_loss))
        print('Train:')
        print('Shredhold:%.2f'%(shredhold))
        print('Accuary:%.2f Precision:%.2f Recall:%.2f F1:%.2f'%(a,p,r,F1))       
        print('Valid:')
        print('Shredhold:%.2f'%(shredhold))
        print('Accuary:%.2f Precision:%.2f Recall:%.2f F1:%.2f'%(v_a,v_p,v_r,v_F1))
        print('Test:')
        print('Accuary:%.2f Precision:%.2f Recall:%.2f F1:%.2f'%(t_a,t_p,t_r,t_F1))

        if v_F1 > best_valid_F1:
            best_shredhold = shredhold
            best_valid_pred,best_test_pred = valid_pred,test_pred
            best_valid_a,best_valid_p,best_valid_r,best_valid_F1 = v_a,v_p,v_r,v_F1
            best_test_a,best_test_p,best_test_r,best_test_F1 = t_a,t_p,t_r,t_F1
            torch.save(model.state_dict(),model_path)
    torch.cuda.empty_cache()
    
    
    with open(result_file,'w') as f:
        f.write('Accuary\tPrecision\tRecall\tF1\n')
        f.write('Valid\n')
        f.write('Shredhold:\t%.2f\n'%(best_shredhold))
        f.write('%.2f\t%.2f\t%.2f\t%.2f\n'%(best_valid_a,best_valid_p,best_valid_r,best_valid_F1))
        f.write('Test\n')
        f.write('Shredhold:\t%.2f\n'%(best_shredhold))
        f.write('%.2f\t%.2f\t%.2f\t%.2f'%(best_test_a,best_test_p,best_test_r,best_test_F1))
    f.close()
    
    with open(pred_file,'w') as f:
        sen = []
        for i in range(len(TestSet.SR_List)):
            if sen != test_sen[i]:
                f.write('\t'.join(test_sen[i])+'\n')
            else:pass
            f.write('\t'.join(test_fc[i][0])+'\n')
            f.write('\t'.join(test_fc[i][1])+'\n')
            if best_test_pred[i] >= best_shredhold and TestSet.SR_List[i] == 1: 
                f.write('%.2f\tRight\t'%(best_test_pred[i])+str(TestSet.SR_List[i])+'\n')
            elif best_test_pred[i] < best_shredhold and TestSet.SR_List[i] == 0:
                f.write('%.2f\tRight\t'%(best_test_pred[i])+str(TestSet.SR_List[i])+'\n')
            else:
                f.write('%.2f\tWrong\t'%(best_test_pred[i])+str(TestSet.SR_List[i])+'\n')
    f.close()

    plt.subplot(1,2,1)
    plt.plot(list(range(1,args.epoch+1)),loss_list)
    plt.plot(list(range(1,args.epoch+1)),train_accuracy,label = 'train')
    plt.plot(list(range(1,args.epoch+1)),valid_accuracy,label = 'valid')
    plt.plot(list(range(1,args.epoch+1)),test_accuracy,label = 'test')
    plt.title('Accuracy')
    plt.subplot(1,2,2)
    plt.plot(list(range(1,args.epoch+1)),loss_list)
    plt.plot(list(range(1,args.epoch+1)),train_F1,label = 'train')
    plt.plot(list(range(1,args.epoch+1)),valid_F1,label = 'valid')
    plt.plot(list(range(1,args.epoch+1)),test_F1,label = 'test')
    plt.title('F1')    
    plt.savefig('./train_process/'+args.encoder+'_'+args.pooling+'_'+args.model+'_'+args.gate+'_'+str(batch_size)+'_'+str(hidden_dim)+'_'+datetime.datetime.now().strftime('%Y%m%d%H%M')+'.jpg')   
            




