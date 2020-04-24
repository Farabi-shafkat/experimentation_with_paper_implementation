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
from save_status import *
import timeit
torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)
torch.backends.cudnn.deterministic=True





def save_model(model, model_name, path,epoch):
    
    model_path = os.path.join(path, '%s%d.pth' % (model_name,epoch))
   # model_path='far.pth'
    torch.save(model.state_dict(), model_path)

def train_phase(train_dataloader, optimizer, epoch,l1,l2,action_criterions):
    model_CNN.train()
    #model_avg.train()
    model_avg_fc.train()
    model_reg.train()
    model_class.train()
   
    iteration = 0
    ret=0
    #print(train_dataloader,"Ssssssssssss............................")
    for data in train_dataloader:
        #print(data,"Ssssssssssss............................")
        #print(data["action"],"....................")
        with torch.no_grad():
            true_final_score = data['label_final_score'].unsqueeze_(1).type(torch.FloatTensor).cuda()
            video = data['video'].transpose_(1, 2).cuda()

            batch_size_now, C, frames, H, W = video.shape
            clip_feats = torch.Tensor([]).cuda()
            sum_feats = None
            #clip_feats.unsqueeze_(0)
            #truePos = data['action']['position'].cuda()
            #trueTw = data['action']['tw_no'].cuda()
            #trueRot = data['action']['rotation_type'].cuda()
            #trueSS = data['action']['ss_no'].cuda()
            #trueArm = data['action']['armstand'].cuda()
            #c1,c2,c3,c4,c5=action_criterions
            divide_by=0;
        for i in np.arange(0, frames - 17, 16):
            clip = video[:, :, i:i + 16, :, :]
            #print(clip.shape," clip shape")
            clip_feats_temp = model_CNN(clip)
            
            #print(clip_feats_temp.shape," clip feat temp shape")
            #print(clip_feats.shape," clip feat  shape")
            
            #clip_feats_temp.transpose_(0, 1)
            
            if sum_feats is None:
                sum_feats=clip_feats_temp
            else:
                sum_feats= sum_feats+clip_feats_temp
            divide_by+=1
            # clip_feats_temp.unsqueeze_(0)
            #clip_feats = torch.cat((clip_feats, clip_feats_temp), 0)
           

            #clip_feats = torch.sum(clip_feats, dim=0)/clip_feats.shape[0]
        clip_feats = sum_feats/divide_by
        features =model_avg_fc(clip_feats) # clip feats has the cip features in axis 1
        pred_final_score=model_reg(features)
        pred_final_score=pred_final_score.view(batch_size_now,-1)       
        aqa_loss = l2(pred_final_score, true_final_score)#+l1(pred_final_score, true_final_score)


        #pos,tw,rot,ss,arm=model_class(features)
        
        #pos=pos.view(batch_size_now,-1)
        #tw=tw.view(batch_size_now,-1)
        #rot=rot.view(batch_size_now,-1)
        
        #ss=ss.view(batch_size_now,-1)
        
        #arm=arm.view(batch_size_now,-1)
        #action_loss = c1(pos,truePos)+c2(tw,trueTw)+c3(rot,trueRot)+c4(ss,trueSS)+c5(arm,trueArm)

 
        loss = aqa_loss*alpha #+action_loss*beta #+caption_loss*gamma
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            if iteration % 50 == 0:
                loss_dic={
                    "mode":"train",
                    "action_quality":aqa_loss.item(),
                    "action_recognition_loss":0,#action_loss.item(),
                    "loss":loss.item()
                }
                save_logs(epoch,iteration,loss_dic)
                print('Epoch: ', epoch, ' Iter: ', iteration, ' Loss: ', loss, end="")
                print(' ')
            iteration += 1
            ret+=aqa_loss.item()
    return ret/iteration

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
            sum_feats = None
            divide_by=0
            for i in np.arange(0, frames - 17, 16):
                clip = video[:, :, i:i + 16, :, :]
            
                clip_feats_temp = model_CNN(clip)

                #clip_feats_temp.unsqueeze_(0)
                #clip_feats_temp.transpose_(0, 1)
                #clip_feats = torch.cat((clip_feats, clip_feats_temp), 0)
                if sum_feats is None:
                    sum_feats=clip_feats_temp
                else:
                    sum_feats= sum_feats+clip_feats_temp
                divide_by+=1
            clip_feats = sum_feats/divide_by                  # torch.sum(clip_feats, dim=0)/clip_feats.shape[0]
            features =model_avg_fc(clip_feats) # clip feats has the cip features in axis 1
            pred_final_score=model_reg(features)
            pred_final_score=pred_final_score.view(batch_size_now,-1)  
          #  print(pred_final_score.shape," ",pred_final_score) 
           
            #if iteration%50==0:
                #print('iteration :',iteration)
                
                #print('pred scores ',pred_scores )
           # pos,tw,rot,ss,arm=model_class(features)
        
            #pos=pos.view(batch_size_now,-1)
            #tw=tw.view(batch_size_now,-1)
            #rot=rot.view(batch_size_now,-1)
        
            #ss=ss.view(batch_size_now,-1)
        
            #arm=arm.view(batch_size_now,-1)
            
            #action_loss = c1(pos,truePos)+c2(tw,trueTw)+c3(rot,trueRot)+c4(ss,trueSS)+c5(arm,trueArm)
            aqa_loss = l2(pred_final_score, true_final_score)#+l1(pred_final_score, true_final_score)
            pred_scores.extend(pred_final_score.cpu().data.numpy())
            iteration+=1 
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
        save_logs(epoch,iteration,loss_dic)
        return aqa_loss_acc/iteration
        #print('Predicted scores: ', pred_scores)
       # print('True scores: ', true_scores)
        #print('Correlation: ', rho)
        


def main(init_epoch):
    parameter_list=list(model_CNN.parameters())+list(model_avg_fc.parameters())+list(model_reg.parameters())#+list(model_class.parameters())
    
    parameters_to_optimize = parameter_list
   # print(len(parameter_list)," $$$$$4")
    #exit()
   # optimizer = optim.Adam(parameters_to_optimize,lr=0.01)
    #torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
   # optimizer = optim.Adadelta(parameters_to_optimize,lr=0.0001,weight_decay=0.5)
    optimizer = optim.Adam(parameters_to_optimize,lr=0.0001)
    l1 =nn.L1Loss()
    l2 =nn.MSELoss()
    c1=nn.CrossEntropyLoss()
    c2=nn.CrossEntropyLoss()
    c3=nn.CrossEntropyLoss()
    c4=nn.CrossEntropyLoss()
    c5=nn.CrossEntropyLoss()
    
    action_criterions=(c1,c2,c3,c4,c5)
    train_dataset = VideoDataset('train')
    test_dataset = VideoDataset('test')
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    grph = graph()

    for epoch in range(init_epoch,num_epochs):
        
        start = timeit.default_timer()
        
        print('-------------------------------------------------------------------------------------------------------')
        
        #train_phase(train_dataloader, optimizer, criterion, epoch)
        tr_loss=train_phase(train_dataloader, optimizer, epoch,l1,l2,action_criterions)
        ts_loss=test_phase(test_dataloader,epoch,l1,l2,action_criterions)
        print(" average training loss:{} , average test loss:{}".format(tr_loss,ts_loss))
        grph.update_graph(tr_loss,ts_loss)
         #if epoch == 0:  # save models every 5 epochs
        grph.draw_and_save()
       # if epoch%4==0 or epoch==num_epochs-1 :
        save_model(model_CNN, 'model_CNN', model_saving_dir,epoch)
        save_model(model_avg_fc, 'model_avg_fc', model_saving_dir,epoch)
        save_model(model_reg, 'model_reg', model_saving_dir,epoch)
        save_model(model_class, 'model_class', model_saving_dir,epoch)
        stop = timeit.default_timer()
        print("time taken each epoch {} seconds".format(stop-start))
    grph.draw_and_save()





if __name__ == '__main__':
    # loading the altered C3D (ie C3D upto before fc-6)
   # os.environ["CUDA_VISIBLE_cudaS"] = '0'
   # set_start_time1(str(datetime.today().strftime('%Y-%m-%d-%H:%M:%S')))
    in_colab = True
    model_CNN_pretrained_dict = None
    model_avg_FC_pretrained_dict = None
    model_class_pretrained_dict = None
    model_reg_pretrained_dict = None
    init_epoch=None
    print(torch.cuda.is_available(),"started" )
    if in_colab==False:
        model_CNN_pretrained_dict = torch.load('/content/c3d.pickle')
        init_epoch=0
    
    
    else :
        log_save_directory=os.path.join(google_drive_dir,log_save_directory)
        graph_save_directory=os.path.join(google_drive_dir,graph_save_directory)
        model_saving_dir=os.path.join(google_drive_dir,model_saving_dir)
       # main_datasets_dir="/content"
        init_epoch=load_status()
        if init_epoch ==0:
             model_CNN_pretrained_dict = torch.load('/content/c3d.pickle')
        else:
            init_epoch+=1
            model_CNN_pretrained_dict = torch.load('experimental_models/model_CNN{}.pth'.format(init_epoch))
            model_avg_FC_pretrained_dict = torch.load('experimental_models/model_avg_fc{}.pth'.format(init_epoch))
            model_reg_pretrained_dict = torch.load('experimental_models/model_reg{}.pth'.format(init_epoch))
            model_class_pretrained_dict = torch.load('experimental_models/model_class{}.pth'.format(init_epoch))

    model_CNN = C3D()
    model_CNN_dict = model_CNN.state_dict()
   
    model_CNN_pretrained_dict = {k: v for k, v in model_CNN_pretrained_dict.items() if k in model_CNN_dict}
    

   # print(model_CNN_pretrained_dict.keys())
    model_CNN_dict.update(model_CNN_pretrained_dict)
    model_CNN.load_state_dict(model_CNN_dict)
    model_CNN = model_CNN.cuda()

   
    model_avg_fc = AVG_FC()
    if init_epoch!=0:
        model_avg_fc.load_state_dict(model_avg_FC_pretrained_dict)
    model_avg_fc  = model_avg_fc.cuda()

    model_reg =linearRegression()
    if init_epoch!=0:
        model_reg.load_state_dict(model_reg_pretrained_dict)
    model_reg =model_reg.cuda()
    
    model_class=classficatiion()
    if init_epoch!=0:
        model_class.load_state_dict(model_class_pretrained_dict)
    model_class =model_class.cuda()

    main(init_epoch)