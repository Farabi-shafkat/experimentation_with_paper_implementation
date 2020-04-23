import os
import torch
from torch.utils.data import DataLoader
from data_loader import VideoDataset
import numpy as np
import random
from models.C3D import C3D
import scipy.stats as stats
import torch.optim as optim
import torch.nn as nn
from models.avg_fc import AVG_FC,linearRegression,classficatiion
from datetime import datetime
from opts import *
from make_graph import graph
from save_logs import *
import itertools  

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)
torch.backends.cudnn.deterministic=True







def test_phase(test_dataloader,epoch,l1,l2,action_criterions):
    with torch.no_grad():
        c1,c2,c3,c4,c5=action_criterions
        pred_scores = []
        true_scores = []
        model_CNN.eval()
        #model_avg.train()
        model_avg_fc.eval()
        model_reg.eval()
       # model_class.eval()
        loss_acc=0
        aqa_loss_acc=0
        action_loss_acc=0
        iteration = 0

        for data in test_dataloader:
            true_final_score = data['label_final_score'].unsqueeze_(1).type(torch.FloatTensor).cuda()
            true_scores.extend(data['label_final_score'].data.numpy())
            video = data['video'].transpose_(1, 2).cuda()
            batch_size_now, C, frames, H, W = video.shape
            clip_feats = torch.Tensor([]).cuda()
            #clip_feats.unsqueeze_(0)
            #truePos = data['action']['position'].cuda()
            #trueTw = data['action']['tw_no'].cuda()
            #trueRot = data['action']['rotation_type'].cuda()
            #trueSS = data['action']['ss_no'].cuda()
            #trueArm = data['action']['armstand'].cuda()

            for i in np.arange(0, frames - 17, 16):
                clip = video[:, :, i:i + 16, :, :]
            
                clip_feats_temp = model_CNN(clip)

                clip_feats_temp.unsqueeze_(0)
                #clip_feats_temp.transpose_(0, 1)
                clip_feats = torch.cat((clip_feats, clip_feats_temp), 0)

            clip_feats = torch.sum(clip_feats, dim=0)/clip_feats.shape[0]
            features =model_avg_fc(clip_feats) # clip feats has the cip features in axis 1
            pred_final_score=model_reg(features)
            pred_final_score=pred_final_score.view(batch_size_now,-1)  
          #  print(pred_final_score.shape," ",pred_final_score) 
            pred_scores.extend(pred_final_score.cpu().data.numpy())
            iteration+=1
            for pred,tr_sr in zip(pred_final_score,true_final_score):
                print("predicted final score{} ......Actual final score{} ".format(pred,tr_sr))
            loss1 = l2(pred_final_score, true_final_score)
            loss2 = l1(pred_final_score, true_final_score)
            aqa_loss = loss1 + loss2 
            print("loss {}...l2 loss:{}....l1 loss:{}".format(aqa_loss,loss1,loss2))
            #loss = aqa_loss*alpha+action_loss*beta
            #action_loss_acc+=action_loss.item()
            aqa_loss_acc+=aqa_loss.item()
            #loss_acc+=loss.item()   

        rho, p = stats.spearmanr(pred_scores, true_scores)
        loss_dic={
                "mode":"test",
                "action_quality":aqa_loss_acc/iteration,
                "action_recognition_loss":0,#action_loss_acc/iteration,
                "loss":0,#loss_acc/iteration,       
                "rho" :rho
        }
        # save_logs(epoch,iteration,loss_dic)
        return aqa_loss_acc/iteration
        #print('Predicted scores: ', pred_scores)
       # print('True scores: ', true_scores)
        #print('Correlation: ', rho)
        


def main():
    #parameter_list=list(model_CNN.parameters())+list(model_avg_fc.parameters())+list(model_reg.parameters())+list(model_class.parameters())
    
    #parameters_to_optimize = parameter_list
   # print(len(parameter_list)," $$$$$4")
    #exit()
   # optimizer = optim.Adam(parameters_to_optimize,lr=0.01)
    #torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
    #optimizer = optim.Adadelta(parameters_to_optimize,lr=0.0001,weight_decay=0.5)
    l1 =nn.L1Loss()
    l2 =nn.MSELoss()
    c1=nn.CrossEntropyLoss()
    c2=nn.CrossEntropyLoss()
    c3=nn.CrossEntropyLoss()
    c4=nn.CrossEntropyLoss()
    c5=nn.CrossEntropyLoss()
    
    action_criterions=(c1,c2,c3,c4,c5)
    #train_dataset = VideoDataset('train')
    test_dataset = VideoDataset('test')
    #train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    #grph = graph()

    for epoch in range(1):
        saving_dir = "saved_models"
       
        
        print('-------------------------------------------------------------------------------------------------------')
        
        #train_phase(train_dataloader, optimizer, criterion, epoch)
        #tr_loss=train_phase(train_dataloader, optimizer, epoch,l1,l2,action_criterions)
        ts_loss=test_phase(test_dataloader,epoch,l1,l2,action_criterions)
        #grph.update_graph(tr_loss,ts_loss)
         #if epoch == 0:  # save models every 5 epochs
        #if epoch%5==0:
        #    save_model(model_CNN, 'model_CNN', saving_dir,epoch)
        #    save_model(model_avg_fc, 'model_avg_fc', saving_dir,epoch)
        #    save_model(model_reg, 'model_reg', saving_dir,epoch)
        #    save_model(model_class, 'model_class', saving_dir,epoch)
   # grph.draw_and_save()





if __name__ == '__main__':
    # loading the altered C3D (ie C3D upto before fc-6)
   # os.environ["CUDA_VISIBLE_cudaS"] = '0'
   # set_start_time1(str(datetime.today().strftime('%Y-%m-%d-%H:%M:%S')))
    
    print(torch.cuda.is_available(),"started" )
    model_CNN_pretrained_dict = torch.load('experimental_models/model_CNN_trial4.pth')
    model_avg_FC_pretrained_dict = torch.load('experimental_models/model_avg_fc_trial4.pth')
    model_class_pretrained_dict = torch.load('experimental_models/model_class_trial4.pth')
    model_reg_pretrained_dict = torch.load('experimental_models/model_reg_trial4.pth')

    model_CNN = C3D()
    model_CNN_dict = model_CNN.state_dict()
    model_CNN_pretrained_dict = {k: v for k, v in model_CNN_pretrained_dict.items() if k in model_CNN_dict}
   # print(model_CNN_pretrained_dict.keys())
    model_CNN_dict.update(model_CNN_pretrained_dict)
    model_CNN.load_state_dict(model_CNN_dict)
    model_CNN = model_CNN.cuda(0)

   
    model_avg_fc = AVG_FC()
    #model_avg_fc_dict = model_avg_fc.state_dict()
    #model_avg_FC_pretrained_dict = {k: v for k, v in model_CNN_pretrained_dict.items() if k in model_avg_FC_dict}
   # print(model_CNN_pretrained_dict.keys())
    #model_CNN_dict.update(model_CNN_pretrained_dict)
    model_avg_fc.load_state_dict(model_avg_FC_pretrained_dict)

    model_avg_fc  = model_avg_fc.cuda(0)

    model_reg =linearRegression()
    model_reg.load_state_dict(model_reg_pretrained_dict)
    model_reg =model_reg.cuda(0)
    
    model_class=classficatiion()
    model_class.load_state_dict(model_class_pretrained_dict)
    model_class =model_class.cuda(0)

    main()