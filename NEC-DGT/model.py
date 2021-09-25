# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 10:27:44 2018

@author: gxjco
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os
from utils import read_data
from sklearn.metrics import r2_score,mean_squared_error
from scipy.stats import pearsonr,spearmanr

def process_edge(Ra):
            for i in range(len(Ra)):
                for j in range(len(Ra[i])):
                    for k in range(len(Ra[i][j])): 
                         Ra[i][j][k]=Ra[i][j][k]
            return Ra
         
def process_node(O):
            for i in range(len(O)):
                for j in range(len(O[i])):
                    for k in range(len(O[i][j])):
                        O[i][j][k]=O[i][j][k]                      
            return O         
         
def node_ACC(O,O_t):
            #Ra=O.reshape(O.shape[0],O.shape[2])
            #Ra_t=Ra_t.reshape(O_t.shape[0],O_t.shape[2])
            count=0
            for i in range(O.shape[0]):
                for j in range(O.shape[2]):
                    #Ra_t[i,1,j]=Ra_t[i,1,j]
                    #Ra_t[i,0,j]=Ra_t[i,0,j]
                    if np.argmax(O_t[i,:,j])==np.argmax(O[i,:,j]):
                           count+=1
            return float(count/(O.shape[0]*O.shape[2]))
        
def top_ACC(Ra,Ra_t):
            Ra=Ra.reshape(Ra.shape[0],Ra.shape[2])
            Ra_t=Ra_t.reshape(Ra_t.shape[0],Ra_t.shape[2])
            count=0
            for i in range(len(Ra)):
                for j in range(len(Ra[i])):
                    if Ra_t[i][j]>5 and Ra[i][j]>5:
                        count+=1
                    if Ra_t[i][j]<5 and Ra[i][j]<5:
                        count+=1
            return float(count/(Ra.shape[0]*Ra.shape[1]))
        
def mse(label,real):
            mse=0
            for i in range(label.shape[0]):
               score=mean_squared_error(label[i,0,:],real[i,0,:])
               mse+=score
            return mse/label.shape[0]
         
def r2(label,real):
            r2=0
            for i in range(label.shape[0]):
               score=r2_score(label[i,0,:],real[i,0,:])
               r2+=score
            return r2/label.shape[0]

def pear(label,real):
            p=0
            for i in range(label.shape[0]):
               score=pearsonr(label[i,0,:],real[i,0,:])[0]
               p+=score
            return p/label.shape[0]
        
def spear(label,real):
            sp=0
            for i in range(label.shape[0]):
               score=spearmanr(label[i,0,:],real[i,0,:])[0]
               sp+=score
            return sp/label.shape[0]
        
class graph2graph(object):
    def __init__(self, sess,Ds,No,Nr, Dr,Dx,De_o,De_r,Mini_batch, checkpoint_dir,epoch,Ds_inter,Dr_inter):
        self.sess = sess
        self.Ds = Ds
        self.No = No
        self.Nr =Nr
        self.Dr = Dr
        self.Ds_inter=Ds_inter
        self.Dr_inter=Dr_inter
        self.Dx=Dx
        self.De_o=De_o
        self.De_r = De_r
        self.mini_batch_num = Mini_batch
        self.epoch=epoch
        # batch normalization : deals with poor initialization helps gradient flow
        self.checkpoint_dir = checkpoint_dir
        self.build_model()
        
    def variable_summaries(self,var,idx):
         """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
         with tf.name_scope('summaries_'+str(idx)):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
            
    def build_model(self):
       self.O_1 = tf.placeholder(tf.float32, [self.mini_batch_num,self.Ds,self.No], name="O")
       self.O_target=tf.placeholder(tf.float32, [self.mini_batch_num,self.Ds,self.No], name="O_target")
       # Relation Matrics R=<Rr,Rs,Ra>
       self.Rr = tf.placeholder(tf.float32, [self.mini_batch_num,self.No,self.Nr], name="Rr")
       self.Rs = tf.placeholder(tf.float32, [self.mini_batch_num,self.No,self.Nr], name="Rs")
       self.Ra_1 = tf.placeholder(tf.float32, [self.mini_batch_num,self.Dr,self.Nr], name="Ra")
       self.Ra_target = tf.placeholder(tf.float32, [self.mini_batch_num,self.Dr,self.Nr], name="Ra_target")
       # External Effects
       self.X = tf.placeholder(tf.float32, [self.mini_batch_num,self.Dx,self.No], name="X")
       
       #step1:
       # marshalling function  
       self.B_1=self.m(self.O_1,self.Rr,self.Rs,self.Ra_1)
       # updating the node state
       self.E_O_1=self.phi_E_O_1(self.B_1)  
       self.C_O_1=self.a_O(self.E_O_1,self.Rr,self.O_1)
       self.O_2,self.O_logits2=self.phi_U_O_1(self.C_O_1) 
       #updating the edge 
       self.E_R_1=self.phi_E_R_1(self.B_1)  #add a constrain to make edge1-2==edge 2-1
       self.C_R_1=self.a_R(self.E_R_1,self.Ra_1)
       self.Ra_2=self.phi_U_R_1(self.C_R_1)  
       
       
       
       #step2:
       # marshalling function, m(G)=B, G=<O,R>  
       self.B_2=self.m(self.O_2,self.Rr,self.Rs,self.Ra_2)
       # updating the node state
       self.E_O_2=self.phi_E_O_2(self.B_2)  
       self.C_O_2=self.a_O(self.E_O_2,self.Rr,self.O_1)
       self.O_3,self.O_logits3=self.phi_U_O_2(self.C_O_2) 
       #updating the edge 
       self.E_R_2=self.phi_E_R_2(self.B_2)  #add a constrain to make edge1-2==edge 2-1
       self.C_R_2=self.a_R(self.E_R_2,self.Ra_1)
       self.Ra_3=self.phi_U_R_2(self.C_R_2)
       
        # loss 
       self.loss_node_mse=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.O_target, logits=self.O_logits3,dim=1))
       self.loss_edge_mse=tf.reduce_mean(tf.reduce_mean(tf.square(self.Ra_3-self.Ra_target),[1,2]))
       self.loss_map=self.map_loss(5)
  
       self.loss_E_O = 0.001*tf.nn.l2_loss(self.E_O_1)+0.001*tf.nn.l2_loss(self.E_O_2)#regulization
       self.loss_E_R = 0.001*tf.nn.l2_loss(self.E_R_1)+0.001*tf.nn.l2_loss(self.E_R_2)#regulization
       
       
       params_list=tf.global_variables()
       for i in range(len(params_list)):
            self.variable_summaries(params_list[i],i)
       self.loss_para=0
       for i in params_list:
          self.loss_para+=0.001*tf.nn.l2_loss(i); 
          
       tf.summary.scalar('node_mse',self.loss_node_mse)
       tf.summary.scalar('edge_mse',self.loss_edge_mse)
       tf.summary.scalar('map_mse',self.loss_map) 
       
       t_vars = tf.trainable_variables()
       self.vars = [var for var in t_vars]
       
       self.saver = tf.train.Saver()


    def m(self,O,Rr,Rs,Ra):
     return tf.concat([tf.matmul(O,Rr),tf.matmul(O,Rs),Ra],1);

    def phi_E_O_1(self,B):
     with tf.variable_scope("phi_E_O1") as scope:  
      h_size=50;
      B_trans=tf.transpose(B,[0,2,1]);
      B_trans=tf.reshape(B_trans,[self.mini_batch_num*self.Nr,(2*self.Ds+self.Dr)]);
  
      w1 = tf.Variable(tf.truncated_normal([(2*self.Ds+self.Dr), h_size], stddev=0.1), name="r1_w1o", dtype=tf.float32);
      b1 = tf.Variable(tf.zeros([h_size]), name="r1_b1o", dtype=tf.float32);
      h1 = tf.nn.relu(tf.matmul(B_trans, w1) + b1);
  
      w2 = tf.Variable(tf.truncated_normal([h_size, h_size], stddev=0.1), name="r1_w2o", dtype=tf.float32);
      b2 = tf.Variable(tf.zeros([h_size]), name="r1_b2o", dtype=tf.float32);
      h2 = tf.nn.relu(tf.matmul(h1, w2) + b2);
  
      w5 = tf.Variable(tf.truncated_normal([h_size, self.De_o], stddev=0.1), name="r1_w5o", dtype=tf.float32);
      b5 = tf.Variable(tf.zeros([self.De_o]), name="r1_b5o", dtype=tf.float32);
      h5 = tf.matmul(h2, w5) + b5;
  
      h5_trans=tf.reshape(h5,[self.mini_batch_num,self.Nr,self.De_o]);
      h5_trans=tf.transpose(h5_trans,[0,2,1]);
      return(h5_trans);
      
    def phi_E_O_2(self,B):
     with tf.variable_scope("phi_E_O2") as scope:  
      h_size=50;
      B_trans=tf.transpose(B,[0,2,1]);
      B_trans=tf.reshape(B_trans,[self.mini_batch_num*self.Nr,(2*self.Ds_inter+self.Dr_inter)]);
  
      w1 = tf.Variable(tf.truncated_normal([(2*self.Ds_inter+self.Dr_inter), h_size], stddev=0.1), name="r2_w1o", dtype=tf.float32);
      b1 = tf.Variable(tf.zeros([h_size]), name="r2_b1o", dtype=tf.float32);
      h1 = tf.nn.relu(tf.matmul(B_trans, w1) + b1);
  
      w2 = tf.Variable(tf.truncated_normal([h_size, h_size], stddev=0.1), name="r2_w2o", dtype=tf.float32);
      b2 = tf.Variable(tf.zeros([h_size]), name="r2_b2o", dtype=tf.float32);
      h2 = tf.nn.relu(tf.matmul(h1, w2) + b2);
  
      w5 = tf.Variable(tf.truncated_normal([h_size, self.De_o], stddev=0.1), name="r2_w5o", dtype=tf.float32);
      b5 = tf.Variable(tf.zeros([self.De_o]), name="r2_b5o", dtype=tf.float32);
      h5 = tf.matmul(h2, w5) + b5;
  
      h5_trans=tf.reshape(h5,[self.mini_batch_num,self.Nr,self.De_o]);
      h5_trans=tf.transpose(h5_trans,[0,2,1]);
      return(h5_trans);      
  
    def a_O(self,E,Rr,O):
       E_bar=tf.matmul(E,tf.transpose(Rr,[0,2,1]));
       return (tf.concat([O,E_bar,self.X],1));

    def phi_U_O_1(self,C):
     with tf.variable_scope("phi_U_O1") as scope:          
       h_size=50;
       C_trans=tf.transpose(C,[0,2,1]);
       C_trans=tf.reshape(C_trans,[self.mini_batch_num*self.No,(self.Ds+self.De_o+self.Dx)]);
       w1 = tf.Variable(tf.truncated_normal([(self.Ds+self.De_o+self.Dx), h_size], stddev=0.1), name="o1_w1o", dtype=tf.float32);
       b1 = tf.Variable(tf.zeros([h_size]), name="o1_b1o", dtype=tf.float32);
       h1 = tf.nn.relu(tf.matmul(C_trans, w1) + b1);
       w2 = tf.Variable(tf.truncated_normal([h_size, self.Ds_inter], stddev=0.1), name="o1_w2o", dtype=tf.float32);
       b2 = tf.Variable(tf.zeros([self.Ds_inter]), name="o1_b2o", dtype=tf.float32);
       h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
       h2_trans=tf.reshape(h2,[self.mini_batch_num,self.No,self.Ds_inter]);
       h2_trans_logits=tf.transpose(h2_trans,[0,2,1]);
       h2_soft=tf.nn.softmax(h2_trans_logits,dim=1)
       return h2_soft,h2_trans_logits
       
    def phi_U_O_2(self,C):
     with tf.variable_scope("phi_U_O2") as scope:          
       h_size=50;
       C_trans=tf.transpose(C,[0,2,1]);
       C_trans=tf.reshape(C_trans,[self.mini_batch_num*self.No,(self.Ds+self.De_o+self.Dx)]);
       w1 = tf.Variable(tf.truncated_normal([(self.Ds+self.De_o+self.Dx), h_size], stddev=0.1), name="o2_w1o", dtype=tf.float32);
       b1 = tf.Variable(tf.zeros([h_size]), name="o2_b1o", dtype=tf.float32);
       h1 = tf.nn.relu(tf.matmul(C_trans, w1) + b1);
       w2 = tf.Variable(tf.truncated_normal([h_size, self.Ds], stddev=0.1), name="o2_w2o", dtype=tf.float32);
       b2 = tf.Variable(tf.zeros([self.Ds]), name="o2_b2o", dtype=tf.float32);
       h2 = tf.matmul(h1, w2) + b2
       h2_trans=tf.reshape(h2,[self.mini_batch_num,self.No,self.Ds]);
       h2_trans_logits=tf.transpose(h2_trans,[0,2,1]);
       h2_soft=tf.nn.softmax(h2_trans_logits,dim=1)
       return h2_soft,h2_trans_logits
   
    def phi_E_R_1(self,B):
     with tf.variable_scope("phi_E_R1") as scope:          
       h_size=100;
       B_trans=tf.transpose(B,[0,2,1]);
       B_trans=tf.reshape(B_trans,[self.mini_batch_num*self.Nr,(2*self.Ds+self.Dr)]);
  
       w1_1 = tf.Variable(tf.truncated_normal([(self.Ds), h_size], stddev=0.1), name="r1_w1r", dtype=tf.float32);
       w1_2 = tf.Variable(tf.truncated_normal([(self.Dr), h_size], stddev=0.1), name="r1_w1r", dtype=tf.float32);
       w1=tf.concat([w1_1,w1_1,w1_2],0)
       #w1 = tf.Variable(tf.truncated_normal([(2*self.Ds+self.Dr), h_size], stddev=0.1), name="r_w1r", dtype=tf.float32);
       b1 = tf.Variable(tf.zeros([h_size]), name="r1_b1r", dtype=tf.float32);
       h1 = tf.nn.relu(tf.matmul(B_trans, w1) + b1);
  
       w2 = tf.Variable(tf.truncated_normal([h_size, self.De_r], stddev=0.1), name="r1_w2r", dtype=tf.float32);
       b2 = tf.Variable(tf.zeros([self.De_r]), name="r1_b2r", dtype=tf.float32);
       h2 = tf.matmul(h1, w2) + b2;
  
       h2_trans=tf.reshape(h2,[self.mini_batch_num,self.Nr,self.De_r]);
       h2_trans=tf.transpose(h2_trans,[0,2,1]);
       h2_trans_bar=tf.matmul(h2_trans,tf.transpose(self.Rr,[0,2,1]));
       
       effects=tf.matmul(h2_trans_bar,self.Rr)+tf.matmul(h2_trans_bar,self.Rs)
       return effects 
       
    def phi_E_R_2(self,B):
     with tf.variable_scope("phi_E_R2") as scope:          
       h_size=100;
       B_trans=tf.transpose(B,[0,2,1]);
       B_trans=tf.reshape(B_trans,[self.mini_batch_num*self.Nr,(2*self.Ds_inter+self.Dr_inter)]);
  
       w1_1 = tf.Variable(tf.truncated_normal([(self.Ds_inter), h_size], stddev=0.1), name="r2_w1r", dtype=tf.float32);
       w1_2 = tf.Variable(tf.truncated_normal([(self.Dr_inter), h_size], stddev=0.1), name="r2_w1r", dtype=tf.float32);
       w1=tf.concat([w1_1,w1_1,w1_2],0)
       #w1 = tf.Variable(tf.truncated_normal([(2*self.Ds+self.Dr), h_size], stddev=0.1), name="r_w1r", dtype=tf.float32);
       b1 = tf.Variable(tf.zeros([h_size]), name="r2_b1r", dtype=tf.float32);
       h1 = tf.nn.relu(tf.matmul(B_trans, w1) + b1);
  
       w2 = tf.Variable(tf.truncated_normal([h_size, self.De_r], stddev=0.1), name="r2_w2r", dtype=tf.float32);
       b2 = tf.Variable(tf.zeros([self.De_r]), name="r2_b2r", dtype=tf.float32);
       h2 = tf.matmul(h1, w2) + b2;
  
       h2_trans=tf.reshape(h2,[self.mini_batch_num,self.Nr,self.De_r]);
       h2_trans=tf.transpose(h2_trans,[0,2,1]);
       h2_trans_bar=tf.matmul(h2_trans,tf.transpose(self.Rr,[0,2,1]));
       
       effects=tf.matmul(h2_trans_bar,self.Rr)+tf.matmul(h2_trans_bar,self.Rs)
       return effects    
       
    def a_R(self,E,Ra):
      C_R=tf.concat([Ra,E],1)
      return (C_R); 
 
    def phi_U_R_1(self,C_R):
     with tf.variable_scope("phi_U_R1") as scope:          
       h_size=100;
       C_trans=tf.transpose(C_R,[0,2,1]);
       C_trans=tf.reshape(C_trans,[self.mini_batch_num*self.Nr,(self.De_r+self.Dr)]);
       
       w1 = tf.Variable(tf.truncated_normal([(self.De_r+self.Dr), h_size], stddev=0.1), name="o1_w1r", dtype=tf.float32);
       b1 = tf.Variable(tf.zeros([h_size]), name="o1_b1r", dtype=tf.float32);
       h1 = tf.nn.relu(tf.matmul(C_trans, w1) + b1);
       
       w2 = tf.Variable(tf.truncated_normal([h_size, self.Dr_inter], stddev=0.1), name="o1_w2r", dtype=tf.float32);
       b2 = tf.Variable(tf.zeros([self.Dr_inter]), name="o1_b2r", dtype=tf.float32);
       h2=tf.nn.relu(tf.matmul(h1, w2) + b2);
       h2_trans=tf.reshape(h2,[self.mini_batch_num,self.Nr,self.Dr_inter]);          
       return tf.transpose(h2_trans,[0,2,1])
   
    def phi_U_R_2(self,C_R):
     with tf.variable_scope("phi_U_R2") as scope:          
       h_size=100;
       C_trans=tf.transpose(C_R,[0,2,1]);
       C_trans=tf.reshape(C_trans,[self.mini_batch_num*self.Nr,(self.De_r+self.Dr)]);
       w1 = tf.Variable(tf.truncated_normal([(self.De_r+self.Dr), h_size], stddev=0.1), name="o2_w1r", dtype=tf.float32);
       b1 = tf.Variable(tf.zeros([h_size]), name="o2_b1r", dtype=tf.float32);
       h1 = tf.nn.relu(tf.matmul(C_trans, w1) + b1);
       w2 = tf.Variable(tf.truncated_normal([h_size, self.Dr], stddev=0.1), name="o2_w2r", dtype=tf.float32);
       b2 = tf.Variable(tf.zeros([self.Dr]), name="o2_b2r", dtype=tf.float32);
       h2_trans=tf.reshape(tf.nn.relu(tf.matmul(h1, w2) + b2),[self.mini_batch_num,self.Nr,self.Dr]);
       h2_trans=tf.transpose(h2_trans,[0,2,1]);           
       return h2_trans
     
      
    def chebyshev_polynomials(self, Ra,k):
         """Calculate Chebyshev polynomials up to order k."""
         #transofrm to adjacent matrix
         s0=np.zeros((Ra.shape[0],1,self.Nr)).astype(np.float32)
         s0[:,:,0:self.No-1]=1
         S=tf.multiply(s0,Ra)
         for i in range(1,self.No):
              s= np.zeros((Ra.shape[0],1,self.Nr)).astype(np.float32)
              s[:,:,i*(self.No-1):i*(self.No-1)+(self.No-1)]=1
              S=tf.concat([S,tf.multiply(s,Ra)],1) #S is the arrange adjacent vector with related node unmasked
         T=np.zeros((Ra.shape[0],self.Nr,self.No)).astype(np.float32)
         for i in range(self.No):
             t=np.zeros((Ra.shape[0],self.No-1,self.No))
             for j in range(0,i):
                 t[:,j,j]=1
             for j in range(i,self.No-1):
                 t[:,j,j+1]=1  
             T[:,i*(self.No-1):i*(self.No-1)+self.No-1,:]=t
         adj=tf.matmul(S,T)
         
         def normalize_adj(adj):
           """Symmetrically normalize adjacency matrix."""
           rowsum=tf.reduce_sum(adj,2)+0.001*np.ones((adj.shape[0],adj.shape[1])).astype(np.float32)
           power=(-0.5)*np.ones((rowsum.shape[0],rowsum.shape[1])).astype(np.float32)
           d_inv_sqrt =tf.pow(rowsum,power)
           d_mat_inv_sqrt = tf.matrix_diag(d_inv_sqrt)
           a=tf.transpose(tf.matmul(adj,d_mat_inv_sqrt),[0,2,1])
           return tf.matmul(a,d_mat_inv_sqrt)  
              
         adj_normalized =normalize_adj(adj)
         I=np.zeros((adj.shape[0],adj.shape[1],adj.shape[2])).astype(np.float32)
         for i in range(adj.shape[0]):
             I[i]=np.eye(adj.shape[1])
         laplacian = I - adj_normalized
        
         largest_eigval =1.5*np.ones((adj.shape[0],1)).astype(np.float32)       # tf.reduce_max(tf.self_adjoint_eig(laplacian)[0],1)
         eig_=tf.divide(2*np.ones(largest_eigval.shape).astype(np.float32),largest_eigval)        
         eig_=tf.reshape(eig_,[I.shape[0],1,1])
         scaled_laplacian=tf.multiply(tf.tile(eig_, multiples=[1, I.shape[1], I.shape[2]]),laplacian)-I


         t_k=tf.concat([tf.reshape(I,[I.shape[0],1,I.shape[1],I.shape[2]]),tf.reshape(scaled_laplacian,[I.shape[0],1,I.shape[1],I.shape[2]])],axis=1)
         
         def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, s_lap):
             s_lap_new=2 * tf.matmul(s_lap,t_k_minus_one) - t_k_minus_two
             return s_lap_new

         for i in range(2, k):
                 t_k_=chebyshev_recurrence(t_k[:,-1,:,:], t_k[:,-2,:,:], scaled_laplacian)
                 t_k=tf.concat([t_k,tf.reshape(t_k_,[I.shape[0],1,I.shape[1],I.shape[2]])],axis=1)

         return t_k
     
         
    def map_conv(self,theta,Ra,O,k):
          t_k=self.chebyshev_polynomials(Ra,k)
          theta_norm=theta/tf.reduce_sum(theta)
          theta_norm1=tf.tile(theta_norm, multiples=[self.mini_batch_num,1,self.No,self.No])
          O_trans=tf.reshape(tf.transpose(O,[0,2,1]),[self.mini_batch_num,1,self.No,self.Ds])
          O_copy=tf.tile(O_trans,multiples=[1,k,1,1])
          conv=tf.matmul(tf.multiply(theta_norm1,t_k),O_copy)
          loss=tf.matmul(O,tf.reduce_sum(conv,1))
          return tf.reduce_sum(loss)
      
    def map_loss(self,k):
         with tf.variable_scope("map_conv") as scope: 
           theta1 = tf.Variable(tf.truncated_normal([1,k,1,1], stddev=0.1), name="map_theta1", dtype=tf.float32);
           theta2 = tf.Variable(tf.truncated_normal([1,k,1,1], stddev=0.1), name="map_theta2", dtype=tf.float32);
           theta3 = tf.Variable(tf.truncated_normal([1,k,1,1], stddev=0.1), name="map_theta3", dtype=tf.float32);
           loss1=self.map_conv(theta1,self.Ra_1,self.O_1,k)+self.map_conv(theta3,self.Ra_3,self.O_3,k)+self.map_conv(theta2,self.Ra_2,self.O_2,k)
           loss2=tf.sqrt(tf.nn.l2_loss(tf.reshape(theta1,[k]))*2)+tf.sqrt(2*tf.nn.l2_loss(tf.reshape(theta2,[k])))+tf.sqrt(2*tf.nn.l2_loss(tf.reshape(theta3,[k])))
           return loss1+0.1*loss2
       
    def train(self, args):
        optimizer = tf.train.AdamOptimizer(0.0001);
        trainer=optimizer.minimize(self.loss_node_mse+self.loss_edge_mse++0.01*self.loss_map+self.loss_para+self.loss_E_O+self.loss_E_R); #

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        #read data
        node_data_train,node_data_test,node_label_train,node_label_test,Ra_data_train,Ra_data_test,Ra_label_train,Ra_label_test,Rr_data,Rs_data,X_data_train,X_data_test=read_data(self)
  

        max_epoches=self.epoch
        counter=1
        for i in range(max_epoches):
          tr_loss_node=0
          tr_loss_edge=0
          tr_loss_map=0
          O_t=[]
          Ra_t=[]
          for j in range(int(len(node_data_train)/self.mini_batch_num)):
            batch_O=node_data_train[j*self.mini_batch_num:(j+1)*self.mini_batch_num];
            batch_O_target=node_label_train[j*self.mini_batch_num:(j+1)*self.mini_batch_num];
            batch_Ra=Ra_data_train[j*self.mini_batch_num:(j+1)*self.mini_batch_num]
            batch_Ra_target=Ra_label_train[j*self.mini_batch_num:(j+1)*self.mini_batch_num]
            batch_X_train=X_data_train[j*self.mini_batch_num:(j+1)*self.mini_batch_num]
            O_t_batch,Ra_t_batch,tr_loss_part_node,tr_loss_part_edge,tr_loss_part_map,_=self.sess.run([self.O_3,self.Ra_3,self.loss_node_mse,self.loss_edge_mse,self.loss_map,trainer],feed_dict={self.O_1:batch_O,self.O_target:batch_O_target,self.Ra_1:batch_Ra,self.Ra_target:batch_Ra_target,self.Rr:Rr_data[:self.mini_batch_num],self.Rs:Rs_data[:self.mini_batch_num],self.X:batch_X_train});
            tr_loss_node+=tr_loss_part_node
            tr_loss_edge+=tr_loss_part_edge
            tr_loss_map+=tr_loss_part_map
            O_t.append(O_t_batch)
            Ra_t.append(Ra_t_batch)
          
          acc_top=top_ACC(Ra_label_train,np.array(Ra_t).reshape(Ra_label_train.shape[0],Ra_label_train.shape[1],Ra_label_train.shape[2]))
          acc_node=node_ACC(node_label_train,np.array(O_t).reshape(node_label_train.shape[0],node_label_train.shape[1],node_label_train.shape[2]))
          print("Epoch "+str(i+1)+" acc_node: "+str(acc_node)+" acc_top: "+str(acc_top)+ " node loss: "+str(tr_loss_node/(int(len(node_data_train)/self.mini_batch_num)))+" edge loss: "+str(tr_loss_edge/(int(len(node_data_train)/self.mini_batch_num)))+" map MSE: "+str(tr_loss_map/(int(len(node_data_train)/self.mini_batch_num))));
          
          counter+=1
          self.save(args.checkpoint_dir, counter)
    
    def save(self, checkpoint_dir, step):
        model_name = "g2g.model"
        model_dir = "%s" % ('flu')
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s" % ('flu')
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
    
    def test(self,args):        
        
        node_data_train,node_data_test,node_label_train,node_label_test,Ra_data_train,Ra_data_test,Ra_label_train,Ra_label_test,Rr_data,Rs_data,X_data_train,X_data_test=read_data(self)  
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        te_loss_node=0
        te_loss_edge=0
        te_loss_map=0
        O_t=[]
        Ra_t=[]
        '''
        node_data_test=node_data_train
        node_label_test=node_label_train
        Ra_data_test=Ra_data_train
        Ra_label_test=Ra_label_train
        X_data_test=X_data_train
        '''
        for j in range(int(len(node_data_test)/self.mini_batch_num)):
          batch_O=node_data_test[j*self.mini_batch_num:(j+1)*self.mini_batch_num];
          batch_O_target=node_label_test[j*self.mini_batch_num:(j+1)*self.mini_batch_num];
          batch_Ra=Ra_data_test[j*self.mini_batch_num:(j+1)*self.mini_batch_num]
          batch_Ra_target=Ra_label_test[j*self.mini_batch_num:(j+1)*self.mini_batch_num]
          batch_X_test=X_data_test[j*self.mini_batch_num:(j+1)*self.mini_batch_num]
          te_loss_part_node,te_loss_part_edge,te_loss_part_map,Ra_part,O_part=self.sess.run([self.loss_node_mse,self.loss_edge_mse,self.loss_map,self.Ra_3,self.O_3],feed_dict={self.O_1:batch_O,self.O_target:batch_O_target,self.Ra_1:batch_Ra,self.Ra_target:batch_Ra_target,self.Rr:Rr_data[:self.mini_batch_num],self.Rs:Rs_data[:self.mini_batch_num],self.X:batch_X_test});
          te_loss_node+=te_loss_part_node
          te_loss_edge+=te_loss_part_edge
          te_loss_map+=te_loss_part_map 
          O_t.append(O_part)
          Ra_t.append(Ra_part)  
        O_t=np.array(O_t).reshape(-1,node_label_test.shape[1],node_label_test.shape[2])  
        Ra_t=np.array(Ra_t).reshape(-1,Ra_label_test.shape[1],Ra_label_test.shape[2])             
        
        np.save('O_t'+str(node_label_test.shape[2])+'.npy',O_t)
        np.save('O_x'+str(node_label_test.shape[2])+'.npy',node_data_test)
        np.save('O_y'+str(node_label_test.shape[2])+'.npy',node_label_test)
        np.save('Ra_t'+str(node_label_test.shape[2])+'.npy',Ra_t)
        np.save('Ra_x'+str(node_label_test.shape[2])+'.npy',Ra_data_test)
        np.save('Ra_y'+str(node_label_test.shape[2])+'.npy',Ra_label_test)
        #evaluate
        #Ra_t=process_edge(Ra_t)
        #O_t=process_node(O_t)
        
        #print('mse-node: '+str(mse(node_label_test,O_t)))
        print('mse-edge: '+str(mse(Ra_label_test,Ra_t)))
        #print('r2-node: '+str(r2(node_label_test,O_t)))
        print('r2-edge: '+str(r2(Ra_label_test,Ra_t)))
        #print('p-node: '+str(pear(node_label_test,O_t)))
        print('p-edge: '+str(pear(Ra_label_test,Ra_t)))
        #print('sp-node: '+str(spear(node_label_test,O_t)))
        print('sp-edge: '+str(spear(Ra_label_test,Ra_t)))        
        print('node_acc: '+str(node_ACC(node_label_test,O_t)))
        print('topol_acc: '+str(top_ACC(Ra_label_test,Ra_t))) 
        
        #print('mse-node2: '+str(mse(node_label_test,node_data_test)))
        #print('topol_acc2: '+str(top_ACC(Ra_data_test,Ra_label_test)))
        