# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 17:10:36 2018

@author: gxjco
"""
import numpy as np
    
def read_data(self):    
      a=np.ones((self.No,self.No))-np.eye(self.No)
      x=np.load('b_emt_input.npy')
      y=np.load('b_emt_target.npy')
      data=np.zeros((x.shape[0],x.shape[1],x.shape[2],5))
      for i in range(x.shape[0]):
         data[i,:,:,0:4]=x[i,:,:,:]
         data[i,:,:,4]=y[i,:,:]         
      #shuffle
      #s=np.load('order.npy')
      #data=data[s]
      node_x=np.zeros((x.shape[0],self.Ds,x.shape[1]))
      node_y=np.zeros((x.shape[0],self.Ds,x.shape[1]))
      for i in range(x.shape[0]):
        b=sum(data[i,:,:,0]*np.eye(self.No))
        c=sum(data[i,:,:,4]*np.eye(self.No))
        for j in range(self.No):
           node_x[i][int(b[j])][j]=1
           node_y[i][int(c[j])][j]=1
      for i in range(x.shape[0]):
         data[i,:,:,0]=data[i,:,:,0]*a  #diagnal is removed
         data[i,:,:,4]=data[i,:,:,4]*a
      Rr_data=np.zeros((500,self.No,self.Nr),dtype=float);
      Rs_data=np.zeros((500,self.No,self.Nr),dtype=float);
      Ra_data=np.zeros((500,self.Dr,self.Nr),dtype=float);
      Ra_label=np.zeros((500,self.Dr,self.Nr),dtype=float);
      X_data=np.zeros((500,self.Dx,self.No),dtype=float);
      
      cnt=0
      for i in range(self.No):
       for j in range(self.No):
         if(i!=j):
           Rr_data[:,i,cnt]=1.0;
           Rs_data[:,j,cnt]=1.0;
           for k in range(data.shape[0]):
            Ra_data[k,0,cnt]=data[k,i,j,0]
            Ra_label[k,0,cnt]=data[k,i,j,4]
           cnt+=1;   
      for i in range(500):
          for j in range(self.No):
             X_data[i,:,j]=np.array([data[i,0,0,1],data[i,0,0,2],data[i,0,0,3]])
        
      node_data_train=node_x[0:250]
      node_label_train=node_y[0:250]
      node_data_test=node_x[250:500]
      node_label_test=node_y[250:500]
      Ra_data_train=Ra_data[0:250]
      Ra_data_test=Ra_data[250:500]
      Ra_label_train=Ra_label[0:250]
      Ra_label_test=Ra_label[250:500]
      X_data_train=X_data[:250]
      X_data_test=X_data[250:500]
      return node_data_train,node_data_test,node_label_train,node_label_test,Ra_data_train,Ra_data_test,Ra_label_train,Ra_label_test,Rr_data,Rs_data,X_data_train,X_data_test


