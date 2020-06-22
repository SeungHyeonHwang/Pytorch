# -*- coding: utf-8 -*-
"""
Created on Fri May  8 17:24:17 2020

@author: hwang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import os
from model import UNet
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from dataloader import basic_dataloader
from tensorboardX import SummaryWriter
import torchvision

def to_np(t):
    return t.cpu().detach().numpy()

# def lr_scheduler(optimizer, curr_iter):
#     """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
 
#     if curr_iter == 30000:
#         optimizer.param_groups[0]['lr'] *= 0.1
#     if curr_iter == 40000:
#         optimizer.param_groups[0]['lr'] *= 0.1

def createFolder(directory):
    try :
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error : " + directory)
        
def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0) # mini-batch
    
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--mode', type=str, default='train')

    # custom args
    parser.add_argument('--input_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gpu_num', type=int, nargs='+', default=[0])
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed) 
    device = args.device
    
    # Model init
    cuda = torch.device('cuda')
    import torch
     
    #  Returns a bool indicating if CUDA is currently available.
    torch.cuda.is_available()
    #  True
    #  Returns the index of a currently selected device.
    torch.cuda.current_device()
    #  0
    #  Returns the number of GPUs available.
    torch.cuda.device_count()
    #  1
    #  Gets the name of a device.
    print(torch.cuda.get_device_name(0))
    torch.cuda.device(0)
    device = 0
    
    
    # unet = UNet().to(device)
    unet = UNet(n_class=1).to(device=cuda)
    criterion = nn.BCEWithLogitsLoss()    
    learning_rates = 1e-5
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.learning_rate)

    unet.train()
    train_dataloader, val_dataloader = basic_dataloader(args.input_size, args.batch_size, args.num_workers)
    curr_lr = args.learning_rate
    
    


    print("Initializing Training!")
    
    # save_path = 'C:/Users/USER/Desktop/hand/model_save/'
    
    # dir_root = 'C:/Users/USER/Desktop/segmentation/'
    # dir_img = dir_root+ 'data/x/train/'
    # dir_mask = dir_root+ 'data/y/label/'
    
    dir_checkpoint = 'C:/Users/USER/Desktop/segmentation/checkpoints/'
    createFolder(dir_checkpoint)
    
    # load model 
    unet.load_state_dict(torch.load(dir_checkpoint + "unet.pth"))
    
    dir_log = 'C:/Users/USER/Desktop/segmentation/test_0617_3/'
    createFolder(dir_log)
    summary = SummaryWriter(logdir = dir_log)
    
    max_score = 0
    offset = 2001
    for epoch_idx in range(1, args.epochs + 1):
        
        losses, val_losses = 0, 0
        dices, val_dices = 0, 0
        
        for batch_idx, (image, mask) in enumerate(train_dataloader):

           
            x = image.to(device=cuda, dtype=torch.float32)
            y = mask.to(device=cuda, dtype=torch.float32)

            optimizer.zero_grad()
            masks_pred = unet.forward(x)
            loss = criterion(masks_pred, y) 
            loss.backward()
            optimizer.step()                      
            losses += loss
            
            masks_pred = F.sigmoid(masks_pred)  
            dice = dice_coeff(masks_pred, y)
            dices += dice 
            
            # summary.add_scalar('train_loss', loss, epoch_idx)
            # summary.add_scalar('train_dice', dice, epoch_idx)
            

                    
        # if (epoch_idx+1)%10 == 0 :             
        #     plt.figure()
        #     img = np.transpose(to_np(image[0,...]), (1,2,0))
        #     plt.imshow(img, cmap='brg')
            
        #     plt.figure()      
        #     msk = to_np(mask[0,0,:,:])
        #     plt.imshow(msk, cmap='gray')
            
        #     plt.figure()           
        #     pr = to_np(masks_pred[0,0,:,:])
        #     plt.imshow(pr, cmap='gray')                                        
        #     plt.show()  
        
        
            if (batch_idx+1)%(args.log_interval) == 0 : 
                print("Train | Epoch {}/{}  Batch {}/{}  loss {:2.4f}  Dice {:2.4f} ". 
                      format(epoch_idx, args.epochs, (batch_idx+1), len(train_dataloader), 
                             losses/(batch_idx+1), dices/(batch_idx+1)))
            
  

        for vbatch_idx, (val_image, val_mask) in enumerate(val_dataloader):
            
            val_x = val_image.to(device=cuda, dtype=torch.float32)
            val_y = val_mask.to(device=cuda, dtype=torch.float32)            
            val_masks_pred = unet.forward(val_x)
            val_loss = criterion(val_masks_pred, val_y) 
            val_losses += val_loss 
            
            val_masks_pred = F.sigmoid(val_masks_pred)  
            val_dice = dice_coeff(val_masks_pred, val_y)
            val_dices += val_dice      
            
            # ex_val_mask_pred = utils.make_grid(val_mask_pred, normalize=True, scale_each=True)
            
            if vbatch_idx == 0 :
                # img = torchvision.utils.make_grid(val_x[0,...], normalize=True, scale_each=True)
                # label = torchvision.utils.make_grid(val_y[0,...], normalize=True, scale_each=True)
                # prediction = torchvision.utils.make_grid(val_masks_pred[0,...], normalize=True, scale_each=True)
                
                summary.add_image('Result/image', val_x[0,...], epoch_idx+offset)
                summary.add_image('Result/label', val_y[0,...], epoch_idx+offset)
                summary.add_image('Result/pred', val_masks_pred[0,...], epoch_idx+offset)             
                
                # summary.add_image('Result/image', val_x[0,...], epoch_idx)
                # summary.add_image('Result/label', val_y[0,...], epoch_idx)
                # summary.add_image('Result/pred', val_masks_pred[0,...], epoch_idx)
            


        
        
        # summary.add_scalar('dice', val_dice, epoch_idx)      
   # 
    
        # if (batch_idx+1)%(args.log_interval) == 0 : 
        print("Epoch | Epoch {}/{}  loss {:2.4f}  Dice {:2.4f}  val-loss {:2.4f}  val-Dice {:2.4f}". 
              format(epoch_idx, args.epochs, 
                     losses/(batch_idx+1), dices/(batch_idx+1),
                     val_losses/(vbatch_idx+1), val_dices/(vbatch_idx+1),
                     ))
            
        if max_score <= val_dices/(vbatch_idx+1) :
            max_score = val_dices/(vbatch_idx+1)
            print("  max score : %f" %(max_score))
            torch.save(unet.state_dict(), dir_checkpoint+'unet_2000.pth')              

        
        summary.add_scalars('Loss', {'train_loss' : losses/(batch_idx+1), 'val_loss' : val_losses/(vbatch_idx+1)}, epoch_idx+offset)
        summary.add_scalars('Dice', {'train_dice' : dices/(batch_idx+1), 'val_dice' : val_dices/(vbatch_idx+1)}, epoch_idx+offset)
        
        
        # # if (batch_idx+1)%(args.log_interval) == 0 : 
    
        
        # print("Epoch {}/{}  Batch {}/{}  loss {:2.4f}  Dice {:2.4f} ". 
        #       format(epoch_idx, args.epochs, (batch_idx+1), len(train_dataloader), 
        #              losses/(batch_idx+1), dices/(batch_idx+1)))
        
        # if max_score <= dices/(batch_idx+1) :
        #     max_score = dices/(batch_idx+1)
        #     print("  max score : %f" %(max_score))
        #     torch.save(unet.state_dict(), dir_checkpoint+'unet.pth')
                    
    summary.close()

                
            # if (batch_idx+1)%(args.log_interval) == 0 : 
                
            #     dice = 0
            #     val_losses = 0
            #     n = 0
                
            #     for vbatch_idx, (val_image, val_mask) in enumerate(val_dataloader):
                
            #         val_x = val_image.to(device, dtype=torch.float32)
            #         val_y = val_mask.to(device, dtype=torch.float32)
                    
            #         val_masks_pred = unet.forward(val_x)
                    
            #         val_loss = criterion(val_masks_pred, val_y) 
                    
            #         val_losses += val_loss 
            #         dice += dice_coeff(val_masks_pred, val_y)
            #         n += val_mask.size(0)
        
            #     print("Epoch {}/{}  Batch {}/{}  loss {:2.4f}  val loss {:2.4f} Dice {}". 
            #           format(epoch_idx, args.epochs, (batch_idx+1), len(train_dataloader), 
            #                  losses/(batch_idx+1),
            #                  val_losses/n, dice/n))
                
            #     if max_score <= dice/n :
            #         max_score = dice/n
            #         print("max score : %f" %(max_score))
                    
                    
                    
                
                # if min_loss >= losses/(batch_idx+1) :
                #     min_loss = losses/(batch_idx+1)
                #     print("min loss : %f " %(min_loss))
                    
                    # torch.save(unet.state_dict(), dir_checkpoint+'unet.pth')
                        
                
                
                        # plt.figure()
                        # img = to_np(image[0,...])
                        # plt.imshow(np.transpose(img, (1,2,0)), cmap='brg')
                        
                        # plt.figure()      
                        # msk = to_np(mask[0,0,:,:])
                        # plt.imshow(msk, cmap='gray')
                                   
                        # plt.figure()           
                        # pr = to_np(masks_pred[0,1,:,:])
                        # plt.imshow(pr, cmap='gray')                                        
                        # plt.show()            
     