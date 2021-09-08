

def train(net,loss_fn,X,Y,n_iter,LR,testX=None,testY=None):
    """
    sorry, not well segmented.
    you can see how this would be modified to test different nets..!
    """
    
    train_loss = []
    test_loss = []
    optimizer = optim.Adam(net.parameters(), lr =0.001, weight_decay = 1e-4)#, momentum=0.9) 
    
   
    conv2d_list = []
    
    for epoch in range(n_iter):
        
        c = 0
        ct = 0
        
        loss_total = 0
        test_loss_total = 0
    
        #changing learning rate.

        for param_group in optimizer.param_groups:
            param_group['lr'] = LR[epoch]
    
    
        batch_size = 6
        
        for k in range(0,X.shape[0],batch_size):
            
            c += 1
            
            newY = np.zeros((batch_size,2,Lyd,Lxd)) 
            #since big numpy arrays crash kernel, just make a thin one.
            newY[:batch_size,:,:,:] = Y[k:k+batch_size,:,:,:] 
            
            #wait, why does it have a third dimension again..?
            newX = X[k:k+batch_size,:,:,:]
            

            #or you can now do: 
            sample_angle = np.random.randint(0,360)
            
            
            #lbl,angle,cut_ind = rotate_and_crop(newY, angle,144) #don't specify cut_ind
            #imgi,angle,cut_ind = rotate_and_crop(newX,angle,144,cut_ind)

            
            imgi,lbl = rc.do_rotated_random_crop(newX,newY,angle=sample_angle,size=144)
            
            img = normalize99(imgi)  
#            img,img1,img2,img3 = downsample_image(imgi)

        
            X_ = img.astype(np.float32)
            X_ = torch.from_numpy(X_).float().to(device=torch.device('cuda')) 
            
#             X_1 = img1.astype(np.float32)
#             X_1 = torch.from_numpy(X_1).float().to(device=torch.device('cuda'))
            
#             X_2 = img2.astype(np.float32)
#             X_2 = torch.from_numpy(X_2).float().to(device=torch.device('cuda'))
            
#             X_3 = img3.astype(np.float32)
#             X_3 = torch.from_numpy(X_3).float().to(device=torch.device('cuda'))


            out_ = net(X_)#,X_1,X_2,X_3)

            Y_ = torch.tensor(lbl, \
                              requires_grad=True).float().to(device=torch.device('cuda'))

            loss = loss_fn(out_,Y_)
            loss_total += (loss.item())
            
            optimizer.zero_grad() #zero out gradients
            loss.backward()
            optimizer.step()
            
           #visualize_midway(imgi,out_.cpu().detach().numpy(),lbl,epoch,k)
    
        loss_avg = loss_total/c
        train_loss.append(loss_avg)
        
        
        
        #testing, if test data is provided:
        
        
        if testX is not None:
            for kt in range(0,testX.shape[0],batch_size):
                            
                ct += 1

                newYt = np.zeros((batch_size,2,Lyd,Lxd)) 
                #since big numpy arrays crash kernel, just make a thin one.
                newYt[:batch_size,:,:,:] = testY[kt:kt+batch_size,:,:,:] 
                newXt = testX[kt:kt+batch_size,:,:,:]


                #angle = np.random.randint(0,360)
                sample_angle = np.random.randint(0,360)


                #lbl,angle,cut_ind = rotate_and_crop(newY, angle,144) #don't specify cut_ind
                #imgi,angle,cut_ind = rotate_and_crop(newX,angle,144,cut_ind)


                imgit,lblt = rc.do_rotated_random_crop(newXt,newYt,angle=sample_angle,size=144)
       
                imgt = normalize99(imgit)
#                imgt,img1t,img2t,img3t = downsample_image(imgit)

                X_t = imgt.astype(np.float32)
                X_t = torch.from_numpy(X_t).float().to(device=torch.device('cuda')) 

#                 X_1t = img1t.astype(np.float32)
#                 X_1t = torch.from_numpy(X_1t).float().to(device=torch.device('cuda'))

#                 X_2t = img2t.astype(np.float32)
#                 X_2t = torch.from_numpy(X_2t).float().to(device=torch.device('cuda'))

#                 X_3t = img3t.astype(np.float32)
#                 X_3t = torch.from_numpy(X_3t).float().to(device=torch.device('cuda'))


                out_t = net(X_t)#,X_1t,X_2t,X_3t)
                Y_t = torch.tensor(lblt, \
                                  requires_grad=True).float().to(device=torch.device('cuda'))

                # Evaluate mean squared error
                losst = loss_fn(out_t,Y_t)
                test_loss_total += (losst.item())
                
                #visualize_midway(imgit,out_t.cpu().detach().numpy(),lblt,epoch,kt)
                
            test_loss_avg = test_loss_total/ct
            test_loss.append(test_loss_avg)
                
                
        #random visualization to see if it helps

        #if c%100==0:
        
        if epoch%10==0:
            visualize_midway(imgi,out_.cpu().detach().numpy(),lbl,epoch,k)
            if testX is not None:
                visualize_midway(imgit,out_t.cpu().detach().numpy(),lblt,epoch,kt)
                        
        print('EPOCH #:',epoch)
        print('training MSE: %.10f' % loss_avg)
        if testX is not None: print('testing MSE: %.10f' % test_loss_avg)
        print("learning rate was: ",LR[epoch],'\n')
        
        

    if testX is not None:
        return train_loss,test_loss
    
    else: 
        return train_loss#,conv2d_list

    