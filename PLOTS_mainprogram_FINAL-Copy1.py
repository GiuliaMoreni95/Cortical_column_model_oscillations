#!/usr/bin/env python
# coding: utf-8

# In[36]:


#This porgrams generates all the plots (raster, rate) based on the simulation main program


# In[47]:


import numpy as np
import time
import matplotlib.pyplot as plt


# In[48]:


#Importing some info I need for this program
runtime=np.loadtxt("general_files/runtime.txt")*1000 #because I want ms
print(runtime)
nu_ext=np.loadtxt('nu_ext_try_g=5_new.txt')
G=np.loadtxt('general_files/G.txt')


# In[49]:


N_arr = np.loadtxt("general_files/N.txt").reshape(4, 4)
#print(N_arr)
N = N_arr.astype(int)
print(N)
print(np.sum(N))
Ntot=np.sum(N)
#Need this for one of the plots
N1=int(0.0192574218*Ntot) # not included in the calculation of Ntot, Ntot is just the 4 layers. 
N2_3=int(0.291088453*Ntot) # This percentage are computed looking at the numbers of the Allen institute 
N4=int(0.237625904*Ntot)
N5=int(0.17425693*Ntot)
N6= Ntot-N2_3-N4-N5


# In[50]:


#Upload the spikes files
#layer1
S_vip1i=np.array(np.loadtxt('Spikes_files/S_vip1i.txt') )
S_vip1t=np.array(np.loadtxt('Spikes_files/S_vip1t.txt') )*1000

#layer23
S_e23i=np.array(np.loadtxt('Spikes_files/S_e23i.txt') )
S_e23t=np.array(np.loadtxt('Spikes_files/S_e23t.txt') )*1000

S_pv23i=np.array(np.loadtxt('Spikes_files/S_pv23i.txt') )
S_pv23t=np.array(np.loadtxt('Spikes_files/S_pv23t.txt') )*1000

S_sst23i=np.array(np.loadtxt('Spikes_files/S_sst23i.txt') )
S_sst23t=np.array(np.loadtxt('Spikes_files/S_sst23t.txt') )*1000

S_vip23i=np.array(np.loadtxt('Spikes_files/S_vip23i.txt') )
S_vip23t=np.array(np.loadtxt('Spikes_files/S_vip23t.txt') )*1000

#layer4
S_e4i=np.array(np.loadtxt('Spikes_files/S_e4i.txt') )
S_e4t=np.array(np.loadtxt('Spikes_files/S_e4t.txt') )*1000

S_pv4i=np.array(np.loadtxt('Spikes_files/S_pv4i.txt') )
S_pv4t=np.array(np.loadtxt('Spikes_files/S_pv4t.txt') )*1000

S_sst4i=np.array(np.loadtxt('Spikes_files/S_sst4i.txt') )
S_sst4t=np.array(np.loadtxt('Spikes_files/S_sst4t.txt') )*1000

S_vip4i=np.array(np.loadtxt('Spikes_files/S_vip4i.txt') )
S_vip4t=np.array(np.loadtxt('Spikes_files/S_vip4t.txt') )*1000

#layer5
S_e5i=np.array(np.loadtxt('Spikes_files/S_e5i.txt') )
S_e5t=np.array(np.loadtxt('Spikes_files/S_e5t.txt') )*1000

S_pv5i=np.array(np.loadtxt('Spikes_files/S_pv5i.txt') )
S_pv5t=np.array(np.loadtxt('Spikes_files/S_pv5t.txt') )*1000

S_sst5i=np.array(np.loadtxt('Spikes_files/S_sst5i.txt') )
S_sst5t=np.array(np.loadtxt('Spikes_files/S_sst5t.txt') )*1000

S_vip5i=np.array(np.loadtxt('Spikes_files/S_vip5i.txt') )
S_vip5t=np.array(np.loadtxt('Spikes_files/S_vip5t.txt') )*1000

#layer6
S_e6i=np.array(np.loadtxt('Spikes_files/S_e6i.txt') )
S_e6t=np.array(np.loadtxt('Spikes_files/S_e6t.txt') )*1000

S_pv6i=np.array(np.loadtxt('Spikes_files/S_pv6i.txt') )
S_pv6t=np.array(np.loadtxt('Spikes_files/S_pv6t.txt') )*1000

S_sst6i=np.array(np.loadtxt('Spikes_files/S_sst6i.txt') )
S_sst6t=np.array(np.loadtxt('Spikes_files/S_sst6t.txt') )*1000

S_vip6i=np.array(np.loadtxt('Spikes_files/S_vip6i.txt') )
S_vip6t=np.array(np.loadtxt('Spikes_files/S_vip6t.txt') )*1000


# In[51]:


# # #------------------------------------------------------------------------------
# # # Show results (all layer) population activity, raster plots SUBFRACTION
# # #------------------------------------------------------------------------------
def raster_plots_sub(n_activity,S_vip1t,
              S_e4t,S_pv4t,S_sst4t,S_vip4t,
              S_e5t,S_pv5t,S_sst5t,S_vip5t,
              S_e6t,S_pv6t,S_sst6t,S_vip6t,
             S_e23t,S_pv23t,S_sst23t,S_vip23t,
                 S_vip1i,S_e4i,S_pv4i,S_sst4i,S_vip4i,
              S_e5i,S_pv5i,S_sst5i,S_vip5i,
              S_e6i,S_pv6i,S_sst6i,S_vip6i,
             S_e23i,S_pv23i,S_sst23i,S_vip23i):

    fig = plt.figure(figsize=(6,4))
    plt.plot(S_vip1t,S_vip1i+ (4 - 0 - 1) * n_activity,'.', markersize=2,color='orange', label='vip')
    plt.xlabel('time (ms)')
    plt.legend()
    plt.title(('Pops activity - layer 1 ({} neurons/pop)'.format(n_activity)))
    plt.show()

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(16,9))
    ax1.plot(S_e23t,S_e23i+ 3 * n_activity,'.', markersize=2,color='r', label='e')
    ax1.plot(S_pv23t,S_pv23i+ 2 * n_activity,'.', markersize=2,color='b', label='pv')
    ax1.plot(S_sst23t,S_sst23i+ n_activity,'.', markersize=2,color='g', label='sst')
    ax1.plot(S_vip23t,S_vip23i,'.', markersize=2,color='orange', label='vip')
    ax1.set_xlabel('time (ms)')
    ax1.legend()
    ax1.set_title(('Pops activity - layer 2/3 ({} neurons/pop)'.format(n_activity)))

    ax2.plot(S_e4t,S_e4i+ 3 * n_activity,'.', markersize=2,color='r', label='e')
    ax2.plot(S_pv4t,S_pv4i+ 2* n_activity,'.', markersize=2,color='b', label='pv')
    ax2.plot(S_sst4t,S_sst4i+ n_activity,'.', markersize=2,color='g', label='sst')
    ax2.plot(S_vip4t,S_vip4i,'.', markersize=2,color='orange', label='vip')
    ax2.set_title(('Pops activity - layer 4 ({} neurons/pop)'.format(n_activity)))
    ax2.set_xlabel('time (ms)')
    ax2.legend()

    ax3.plot(S_e5t,S_e5i+ 3 * n_activity,'.', markersize=2,color='r', label='e')
    ax3.plot(S_pv5t,S_pv5i+ 2 * n_activity,'.', markersize=2,color='b', label='pv')
    ax3.plot(S_sst5t,S_sst5i+ 1 * n_activity,'.', markersize=2,color='g', label='sst')
    ax3.plot(S_vip5t,S_vip5i,'.', markersize=2,color='orange', label='vip')
    ax3.set_ylim(0,40)
    ax3.set_title(('Pops activity - layer 5 ({} neurons/pop)'.format(n_activity)))
    ax3.set_xlabel('time (ms)')
    ax3.legend()

    ax4.plot(S_e6t,S_e6i+ 3 * n_activity,'.', markersize=2,color='r', label='e')
    ax4.plot(S_pv6t,S_pv6i+ 2 * n_activity,'.', markersize=2,color='b', label='pv')
    ax4.plot(S_sst6t,S_sst6i+ 1 * n_activity,'.', markersize=2,color='g', label='sst')
    ax4.plot(S_vip6t,S_vip6i,'.', markersize=2,color='orange', label='vip')
    ax4.set_ylim(0,40)
    ax4.set_title(('Pops activity - layer 6 ({} neurons/pop)'.format(n_activity)))
    ax4.set_xlabel('time (ms)')
    ax4.legend()

    plt.subplots_adjust(left=0.125,
                        bottom=0.125, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.2, 
                        hspace=0.3)

    plt.show()


# In[52]:


# #------------------------------------------------------------------------------
# # Show results (all layer) population activity, raster plots (recording from all neurons)
#4 different plots, one for each layer
# #------------------------------------------------------------------------------
#In a function
def raster_plots(N,S_vip1t,
              S_e4t,S_pv4t,S_sst4t,S_vip4t,
              S_e5t,S_pv5t,S_sst5t,S_vip5t,
              S_e6t,S_pv6t,S_sst6t,S_vip6t,
             S_e23t,S_pv23t,S_sst23t,S_vip23t,
                 S_vip1i,S_e4i,S_pv4i,S_sst4i,S_vip4i,
              S_e5i,S_pv5i,S_sst5i,S_vip5i,
              S_e6i,S_pv6i,S_sst6i,S_vip6i,
             S_e23i,S_pv23i,S_sst23i,S_vip23i):
    
    i_e23=S_e23i+ N[0][3]+N[0][2]+N[0][1]
    i_pv23=S_pv23i+ N[0][3]+N[0][2]
    i_sst23=S_sst23i+ N[0][3]
    i_vip23=S_vip23i

    i_e4=S_e4i+ N[1][3]+N[1][2]+N[1][1]
    i_pv4=S_pv4i+ N[1][3]+N[1][2]
    i_sst4=S_sst4i+ N[1][3]
    i_vip4=S_vip4i

    i_e5=S_e5i+ N[2][3]+N[2][2]+N[2][1]
    i_pv5=S_pv5i+ N[2][3]+N[2][2]
    i_sst5=S_sst5i+ N[2][3]
    i_vip5=S_vip5i

    i_e6=S_e6i+ N[3][3]+N[3][2]+N[3][1]
    i_pv6=S_pv6i+ N[3][3]+N[3][2]
    i_sst6=S_sst6i+ N[3][3]
    i_vip6=S_vip6i

    fig = plt.figure(figsize=(6,4))
    plt.plot(S_vip1t,S_vip1i,'.', markersize=2,color='orange', label='vip')
    plt.xlabel('time (ms)')
    plt.legend()
    plt.title('Pops activity - layer 1 ')
    plt.show()

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(16,9))
    ax1.plot(S_e23t,i_e23,'.', markersize=2,color='r', label='e')
    ax1.plot(S_pv23t,i_pv23,'.', markersize=2,color='b', label='pv')
    ax1.plot(S_sst23t,i_sst23,'.', markersize=2,color='g', label='sst')
    ax1.plot(S_vip23t,i_vip23,'.', markersize=2,color='orange', label='vip')
    #ax2=ax1.twinx()
    #ax2.plot(x,I,'k')
    ax1.set_xlabel('time (ms)')
    ax1.set_ylabel('neuron index')
    #ax1.set_xlim(1200,2000)
    ax1.legend()
    ax1.set_title('Pops activity - layer 2/3 ')

    ax2.plot(S_e4t,i_e4,'.', markersize=2,color='r', label='e')
    ax2.plot(S_pv4t,i_pv4,'.', markersize=2,color='b', label='pv')
    ax2.plot(S_sst4t,i_sst4,'.', markersize=2,color='g', label='sst')
    ax2.plot(S_vip4t,i_vip4,'.', markersize=2,color='orange', label='vip')
    ax2.set_title('Pops activity - layer 4')
    #ax2.set_ylim(200,210)
    ax2.set_xlabel('time (ms)')
    ax2.set_ylabel('neuron index')
    ax2.legend()

    ax3.plot(S_e5t,i_e5,'.', markersize=2,color='r', label='e')
    ax3.plot(S_pv5t,i_pv5,'.', markersize=2,color='b', label='pv')
    ax3.plot(S_sst5t,i_sst5,'.', markersize=2,color='g', label='sst')
    ax3.plot(S_vip5t,i_vip5,'.', markersize=2,color='orange', label='vip')
    #ax3.set_xlim(1200,2000)
    ax3.set_title('Pops activity - layer 5' )
    ax3.set_xlabel('time (ms)')
    ax3.set_ylabel('neuron index')
    ax3.legend()

    ax4.plot(S_e6t,i_e6,'.', markersize=2,color='r', label='e')
    ax4.plot(S_pv6t,i_pv6,'.', markersize=2,color='b', label='pv')
    ax4.plot(S_sst6t,i_sst6,'.', markersize=2,color='g', label='sst')
    ax4.plot(S_vip6t,i_vip6,'.', markersize=2,color='orange', label='vip')
    #ax4.set_ylim(0,40)
    #ax4.set_xlim(1200,2000)
    ax4.set_title('Pops activity - layer 6' )
    ax4.set_xlabel('time (ms)')
    ax4.set_ylabel('neuron index')
    ax4.legend()

    plt.subplots_adjust(left=0.125,
                        bottom=0.125, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.2, 
                        hspace=0.3)

    plt.show()


# In[53]:


raster_plots(N,S_vip1t,
              S_e4t,S_pv4t,S_sst4t,S_vip4t,
              S_e5t,S_pv5t,S_sst5t,S_vip5t,
              S_e6t,S_pv6t,S_sst6t,S_vip6t,
             S_e23t,S_pv23t,S_sst23t,S_vip23t,
                 S_vip1i,S_e4i,S_pv4i,S_sst4i,S_vip4i,
              S_e5i,S_pv5i,S_sst5i,S_vip5i,
              S_e6i,S_pv6i,S_sst6i,S_vip6i,
             S_e23i,S_pv23i,S_sst23i,S_vip23i)


# In[9]:


#raster plot of all neurons all layers in the same plot
def raster_all(N,N4,N5,N6,S_vip1t,
              S_e4t,S_pv4t,S_sst4t,S_vip4t,
              S_e5t,S_pv5t,S_sst5t,S_vip5t,
              S_e6t,S_pv6t,S_sst6t,S_vip6t,
             S_e23t,S_pv23t,S_sst23t,S_vip23t,
                 S_vip1i,S_e4i,S_pv4i,S_sst4i,S_vip4i,
              S_e5i,S_pv5i,S_sst5i,S_vip5i,
              S_e6i,S_pv6i,S_sst6i,S_vip6i,
             S_e23i,S_pv23i,S_sst23i,S_vip23i):

    
    i_e23=S_e23i+ N[0][3]+N[0][2]+N[0][1]
    i_pv23=S_pv23i+ N[0][3]+N[0][2]
    i_sst23=S_sst23i+ N[0][3]
    i_vip23=S_vip23i

    i_e4=S_e4i+ N[1][3]+N[1][2]+N[1][1]
    i_pv4=S_pv4i+ N[1][3]+N[1][2]
    i_sst4=S_sst4i+ N[1][3]
    i_vip4=S_vip4i

    i_e5=S_e5i+ N[2][3]+N[2][2]+N[2][1]
    i_pv5=S_pv5i+ N[2][3]+N[2][2]
    i_sst5=S_sst5i+ N[2][3]
    i_vip5=S_vip5i

    i_e6=S_e6i+ N[3][3]+N[3][2]+N[3][1]
    i_pv6=S_pv6i+ N[3][3]+N[3][2]
    i_sst6=S_sst6i+ N[3][3]
    i_vip6=S_vip6i
    
    
    f,ax1= plt.subplots(figsize=(16,9))
    ax1.plot(S_e23t,i_e23+N6+N5+N4,'.', markersize=2,color='r', label='e')
    ax1.plot(S_pv23t,i_pv23+N6+N5+N4,'.', markersize=2,color='b', label='pv')
    ax1.plot(S_sst23t,i_sst23+N6+N5+N4,'.', markersize=2,color='g', label='sst')
    ax1.plot(S_vip23t,i_vip23+N6+N5+N4,'.', markersize=2,color='orange', label='vip')

    ax1.plot(S_e4t,i_e4+N6+N5,'.', markersize=2,color='r')
    ax1.plot(S_pv4t,i_pv4+N6+N5,'.', markersize=2,color='b')
    ax1.plot(S_sst4t,i_sst4+N6+N5,'.', markersize=2,color='g')
    ax1.plot(S_vip4t,i_vip4+N6+N5,'.', markersize=2,color='orange')


    ax1.plot(S_e5t,i_e5+N6,'.', markersize=2,color='r')
    ax1.plot(S_pv5t,i_pv5+N6,'.', markersize=2,color='b')
    ax1.plot(S_sst5t,i_sst5+N6,'.', markersize=2,color='g')
    ax1.plot(S_vip5t,i_vip5+N6,'.', markersize=2,color='orange')

    ax1.plot(S_e6t,i_e6,'.', markersize=2,color='r')
    ax1.plot(S_pv6t,i_pv6,'.', markersize=2,color='b')
    ax1.plot(S_sst6t,i_sst6,'.', markersize=2,color='g')
    ax1.plot(S_vip6t,i_vip6,'.', markersize=2,color='orange')
    ax1.set_xlabel('time (ms)', size=20)
    ax1.set_ylabel('neuron index', size=20)
    ax1.legend()
    ax1.set_title('Populations activity in all layers (layers 2/3, 4, 5, 6) ', size=25)

    #FROM LAYER 2/3 TOP to LAYER 6 bottom
    plt.show()


# In[10]:


raster_all(N,N4,N5,N6,S_vip1t,
              S_e4t,S_pv4t,S_sst4t,S_vip4t,
              S_e5t,S_pv5t,S_sst5t,S_vip5t,
              S_e6t,S_pv6t,S_sst6t,S_vip6t,
             S_e23t,S_pv23t,S_sst23t,S_vip23t,
                 S_vip1i,S_e4i,S_pv4i,S_sst4i,S_vip4i,
              S_e5i,S_pv5i,S_sst5i,S_vip5i,
              S_e6i,S_pv6i,S_sst6i,S_vip6i,
             S_e23i,S_pv23i,S_sst23i,S_vip23i)


# In[11]:


#When I have an input to show it in the plot
def raster_plots_input(N,S_vip1t,
              S_e4t,S_pv4t,S_sst4t,S_vip4t,
              S_e5t,S_pv5t,S_sst5t,S_vip5t,
              S_e6t,S_pv6t,S_sst6t,S_vip6t,
             S_e23t,S_pv23t,S_sst23t,S_vip23t,
                 S_vip1i,S_e4i,S_pv4i,S_sst4i,S_vip4i,
              S_e5i,S_pv5i,S_sst5i,S_vip5i,
              S_e6i,S_pv6i,S_sst6i,S_vip6i,
             S_e23i,S_pv23i,S_sst23i,S_vip23i):
    
    
    #Input part
    t1=500
    t2=800
    t3=2100
    t4=2300
    runtime=3000
    y = np.array([0 for i in range(0,t1)])
    y1=np.array([30 for i in range(t1,t2)])
    y2=np.array([0 for i in range(t2,t3)])
    y3=np.array([30 for i in range(t3,t4)])
    y4=np.array([0 for i in range(t4,runtime)])
    I=np.concatenate((y, y1,y2,y3,y4), axis=0)
    
    x = np.array([i for i in range(0,runtime)])

    activation_point=500
    runtime=3000

    y = np.array([0 for i in range(0,activation_point)])
    y1=np.array([30 for i in range(activation_point,runtime)])
    I=np.concatenate((y, y1), axis=0)




    i_e23=S_e23i+ N[0][3]+N[0][2]+N[0][1]
    i_pv23=S_pv23i+ N[0][3]+N[0][2]
    i_sst23=S_sst23i+ N[0][3]
    i_vip23=S_vip23i

    i_e4=S_e4i+ N[1][3]+N[1][2]+N[1][1]
    i_pv4=S_pv4i+ N[1][3]+N[1][2]
    i_sst4=S_sst4i+ N[1][3]
    i_vip4=S_vip4i

    i_e5=S_e5i+ N[2][3]+N[2][2]+N[2][1]
    i_pv5=S_pv5i+ N[2][3]+N[2][2]
    i_sst5=S_sst5i+ N[2][3]
    i_vip5=S_vip5i

    i_e6=S_e6i+ N[3][3]+N[3][2]+N[3][1]
    i_pv6=S_pv6i+ N[3][3]+N[3][2]
    i_sst6=S_sst6i+ N[3][3]
    i_vip6=S_vip6i

    # fig = plt.figure(figsize=(6,4))
    # plt.plot(S_vip1.t/ms,S_vip1.i,'.', markersize=2,color='orange', label='vip')
    # plt.xlabel('time (ms)')
    # plt.legend()
    # plt.title('Pops activity - layer 1 ')
    # plt.show()

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(16,9))
    ax1.plot(S_e23t,i_e23,'.', markersize=2,color='r', label='e')
    ax1.plot(S_pv23t,i_pv23,'.', markersize=2,color='b', label='pv')
    ax1.plot(S_sst23t,i_sst23,'.', markersize=2,color='g', label='sst')
    ax1.plot(S_vip23t,i_vip23,'.', markersize=2,color='orange', label='vip')
    #ax2=ax1.twinx()
    #ax2.plot(x,I,'k')
    ax1.set_xlabel('time (ms)')
    ax1.set_ylabel('neuron index')
    #ax1.set_xlim(1200,2000)
    ax1.legend()
    ax1.set_title('Pops activity - layer 2/3 ')

    ax1 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax1.set_ylabel('Input to e5 (pA)', color='c')  # we already handled the x-label with ax1
    ax1.plot(x, I, color='c')
    ax1.tick_params(axis='y', labelcolor='c')

    ax2.plot(S_e4t,i_e4,'.', markersize=2,color='r', label='e')
    ax2.plot(S_pv4t,i_pv4,'.', markersize=2,color='b', label='pv')
    ax2.plot(S_sst4t,i_sst4,'.', markersize=2,color='g', label='sst')
    ax2.plot(S_vip4t,i_vip4,'.', markersize=2,color='orange', label='vip')
    ax2.set_title('Pops activity - layer 4')
    #ax2.set_xlim(1200,2000)
    ax2.set_xlabel('time (ms)')
    ax2.set_ylabel('neuron index')
    ax2.legend()

    ax2 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Input to e5 (pA)', color='c')  # we already handled the x-label with ax1
    ax2.plot(x, I, color='c')
    ax2.tick_params(axis='y', labelcolor='c')



    ax3.plot(S_e5t,i_e5,'.', markersize=2,color='r', label='e')
    ax3.plot(S_pv5t,i_pv5,'.', markersize=2,color='b', label='pv')
    ax3.plot(S_sst5t,i_sst5,'.', markersize=2,color='g', label='sst')
    ax3.plot(S_vip5t,i_vip5,'.', markersize=2,color='orange', label='vip')
    #ax3.set_xlim(1200,2000)
    ax3.set_title('Pops activity - layer 5' )
    ax3.set_xlabel('time (ms)')
    ax3.set_ylabel('neuron index')
    ax3.legend()

    ax3 = ax3.twinx()  # instantiate a second axes that shares the same x-axis
    ax3.set_ylabel('Input to e5 (pA)', color='c')  # we already handled the x-label with ax1
    ax3.plot(x, I, color='c')
    ax3.tick_params(axis='y', labelcolor='c')




    ax4.plot(S_e6t,i_e6,'.', markersize=2,color='r', label='e')
    ax4.plot(S_pv6t,i_pv6,'.', markersize=2,color='b', label='pv')
    ax4.plot(S_sst6t,i_sst6,'.', markersize=2,color='g', label='sst')
    ax4.plot(S_vip6t,i_vip6,'.', markersize=2,color='orange', label='vip')
    #ax4.set_ylim(0,40)
    #ax4.set_xlim(1200,2000)
    ax4.set_title('Pops activity - layer 6' )
    ax4.set_xlabel('time (ms)')
    ax4.set_ylabel('neuron index')
    ax4.legend()

    ax4 = ax4.twinx()  # instantiate a second axes that shares the same x-axis
    ax4.set_ylabel('Input to e5 (pA)', color='c')  # we already handled the x-label with ax1
    ax4.plot(x, I, color='c')
    ax4.tick_params(axis='y', labelcolor='c')


    plt.subplots_adjust(left=0.125,
                        bottom=0.125, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.3, 
                        hspace=0.3)

    plt.show()


# In[12]:


raster_plots_input(N,S_vip1t,
              S_e4t,S_pv4t,S_sst4t,S_vip4t,
              S_e5t,S_pv5t,S_sst5t,S_vip5t,
              S_e6t,S_pv6t,S_sst6t,S_vip6t,
             S_e23t,S_pv23t,S_sst23t,S_vip23t,
                 S_vip1i,S_e4i,S_pv4i,S_sst4i,S_vip4i,
              S_e5i,S_pv5i,S_sst5i,S_vip5i,
              S_e6i,S_pv6i,S_sst6i,S_vip6i,
             S_e23i,S_pv23i,S_sst23i,S_vip23i)


# In[13]:


#raster plot with input all layers in the same plot
def raster_all_input(N,N4,N5,N6,S_vip1t,
              S_e4t,S_pv4t,S_sst4t,S_vip4t,
              S_e5t,S_pv5t,S_sst5t,S_vip5t,
              S_e6t,S_pv6t,S_sst6t,S_vip6t,
             S_e23t,S_pv23t,S_sst23t,S_vip23t,
                 S_vip1i,S_e4i,S_pv4i,S_sst4i,S_vip4i,
              S_e5i,S_pv5i,S_sst5i,S_vip5i,
              S_e6i,S_pv6i,S_sst6i,S_vip6i,
             S_e23i,S_pv23i,S_sst23i,S_vip23i):

    
    i_e23=S_e23i+ N[0][3]+N[0][2]+N[0][1]
    i_pv23=S_pv23i+ N[0][3]+N[0][2]
    i_sst23=S_sst23i+ N[0][3]
    i_vip23=S_vip23i

    i_e4=S_e4i+ N[1][3]+N[1][2]+N[1][1]
    i_pv4=S_pv4i+ N[1][3]+N[1][2]
    i_sst4=S_sst4i+ N[1][3]
    i_vip4=S_vip4i

    i_e5=S_e5i+ N[2][3]+N[2][2]+N[2][1]
    i_pv5=S_pv5i+ N[2][3]+N[2][2]
    i_sst5=S_sst5i+ N[2][3]
    i_vip5=S_vip5i

    i_e6=S_e6i+ N[3][3]+N[3][2]+N[3][1]
    i_pv6=S_pv6i+ N[3][3]+N[3][2]
    i_sst6=S_sst6i+ N[3][3]
    i_vip6=S_vip6i
    
    #Input part
    t1=500
    t2=800
    t3=2100
    t4=2300
    runtime=3000
    y = np.array([0 for i in range(0,t1)])
    y1=np.array([30 for i in range(t1,t2)])
    y2=np.array([0 for i in range(t2,t3)])
    y3=np.array([30 for i in range(t3,t4)])
    y4=np.array([0 for i in range(t4,runtime)])
    I=np.concatenate((y, y1,y2,y3,y4), axis=0)
    
    
    activation_point=500
    runtime=3000
    x = np.array([i for i in range(0,runtime)])

    y = np.array([0 for i in range(0,activation_point)])
    y1=np.array([30 for i in range(activation_point,runtime)])
    I=np.concatenate((y, y1), axis=0)
    
    f,ax1= plt.subplots(figsize=(16,9))
    ax1.plot(S_e23t,i_e23+N6+N5+N4,'.', markersize=2,color='r', label='e')
    ax1.plot(S_pv23t,i_pv23+N6+N5+N4,'.', markersize=2,color='b', label='pv')
    ax1.plot(S_sst23t,i_sst23+N6+N5+N4,'.', markersize=2,color='g', label='sst')
    ax1.plot(S_vip23t,i_vip23+N6+N5+N4,'.', markersize=2,color='orange', label='vip')

    ax1.plot(S_e4t,i_e4+N6+N5,'.', markersize=2,color='r')
    ax1.plot(S_pv4t,i_pv4+N6+N5,'.', markersize=2,color='b')
    ax1.plot(S_sst4t,i_sst4+N6+N5,'.', markersize=2,color='g')
    ax1.plot(S_vip4t,i_vip4+N6+N5,'.', markersize=2,color='orange')


    ax1.plot(S_e5t,i_e5+N6,'.', markersize=2,color='r')
    ax1.plot(S_pv5t,i_pv5+N6,'.', markersize=2,color='b')
    ax1.plot(S_sst5t,i_sst5+N6,'.', markersize=2,color='g')
    ax1.plot(S_vip5t,i_vip5+N6,'.', markersize=2,color='orange')

    ax1.plot(S_e6t,i_e6,'.', markersize=2,color='r')
    ax1.plot(S_pv6t,i_pv6,'.', markersize=2,color='b')
    ax1.plot(S_sst6t,i_sst6,'.', markersize=2,color='g')
    ax1.plot(S_vip6t,i_vip6,'.', markersize=2,color='orange')
    ax1.set_xlabel('time (ms)', size=20)
    ax1.set_ylabel('neuron index', size=20)
    ax1.legend()
    ax1 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax1.set_ylabel('Input to e5 (pA)', color='c', size=20)  # we already handled the x-label with ax1
    ax1.plot(x, I, color='c')
    ax1.tick_params(axis='y', labelcolor='c')
    ax1.set_title('Populations activity in all layers (layers 2/3, 4, 5, 6) ', size=25)

    #FROM LAYER 2/3 TOP to LAYER 6 bottom
    plt.show()


# In[14]:


raster_all_input(N,N4,N5,N6,S_vip1t,
              S_e4t,S_pv4t,S_sst4t,S_vip4t,
              S_e5t,S_pv5t,S_sst5t,S_vip5t,
              S_e6t,S_pv6t,S_sst6t,S_vip6t,
             S_e23t,S_pv23t,S_sst23t,S_vip23t,
                 S_vip1i,S_e4i,S_pv4i,S_sst4i,S_vip4i,
              S_e5i,S_pv5i,S_sst5i,S_vip5i,
              S_e6i,S_pv6i,S_sst6i,S_vip6i,
             S_e23i,S_pv23i,S_sst23i,S_vip23i)


# In[15]:


print(runtime)


# In[16]:


#ONLY ONE RASTER PLOT WITH INPUT LINE
def raster_one(S_ei,S_pvi,S_ssti,S_vipi,S_et,S_pvt,S_sstt,S_vipt,layer):
    activation_point=300

    x = np.array([i for i in range(0,int(runtime))])
    print(runtime)
    y = np.array([0 for i in range(0,activation_point)])
    y1=np.array([30 for i in range(activation_point,int(runtime))])
    I=np.concatenate((y, y1), axis=0)


    i_e=S_ei+ N[layer][3]+N[layer][2]+N[layer][1]
    i_pv=S_pvi+ N[layer][3]+N[layer][2]
    i_sst=S_ssti+ N[layer][3]
    i_vip=S_vipi

    fig, ax1 = plt.subplots()

    ax1.plot(S_et,i_e,'.', markersize=2,color='r', label='e')
    ax1.plot(S_pvt,i_pv,'.', markersize=2,color='b', label='pv')
    ax1.plot(S_sstt,i_sst,'.', markersize=2,color='g', label='sst')
    ax1.plot(S_vipt,i_vip,'.', markersize=2,color='orange', label='vip')
    #ax2=ax1.twinx()
    #ax2.plot(x,I,'k')
    ax1.set_xlabel('time (ms)')
    ax1.set_ylabel('neuron index')
    #ax1.set_xlim(1200,2000)
    ax1.legend()
    ax1.set_title('Pops activity - layer 2/3 ')

        #f= plt.figure(figsize=(16,9))

    #     plt.plot(x,I,'k')
    #     plt.xlabel('time (ms)',fontsize=15)
    #     plt.ylabel('Input (pA)',fontsize=15)
    #     plt.title('Input to e4',fontsize=15)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis


    ax2.set_ylabel('Input (pA)', color='c')  # we already handled the x-label with ax1
    ax2.plot(x, I, color='c')
    ax2.tick_params(axis='y', labelcolor='c')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


# In[17]:


raster_one(S_e23i,S_pv23i,S_sst23i,S_vip23i,S_e23t,S_pv23t,S_sst23t,S_vip23t,0)


# In[18]:


#Upload the rate files
R_vip1rate=np.array(np.loadtxt('Rate_files/R_vip1rate.txt') )

#layer23
R_e23rate=np.array(np.loadtxt('Rate_files/R_e23rate.txt') )
R_pv23rate=np.array(np.loadtxt('Rate_files/R_pv23rate.txt') )
R_sst23rate=np.array(np.loadtxt('Rate_files/R_sst23rate.txt') )
R_vip23rate=np.array(np.loadtxt('Rate_files/R_vip23rate.txt') )

#layer4
R_e4rate=np.array(np.loadtxt('Rate_files/R_e4rate.txt') )
R_pv4rate=np.array(np.loadtxt('Rate_files/R_pv4rate.txt') )
R_sst4rate=np.array(np.loadtxt('Rate_files/R_sst4rate.txt') )
R_vip4rate=np.array(np.loadtxt('Rate_files/R_vip4rate.txt') )

#layer5
R_e5rate=np.array(np.loadtxt('Rate_files/R_e5rate.txt') )
R_pv5rate=np.array(np.loadtxt('Rate_files/R_pv5rate.txt') )
R_sst5rate=np.array(np.loadtxt('Rate_files/R_sst5rate.txt') )
R_vip5rate=np.array(np.loadtxt('Rate_files/R_vip5rate.txt') )

#layer6
R_e6rate=np.array(np.loadtxt('Rate_files/R_e6rate.txt') )
R_pv6rate=np.array(np.loadtxt('Rate_files/R_pv6rate.txt') )
R_sst6rate=np.array(np.loadtxt('Rate_files/R_sst6rate.txt') )
R_vip6rate=np.array(np.loadtxt('Rate_files/R_vip6rate.txt') ) 


# In[19]:


#Fucntion to compute the average firing rate over all neuron with a sliding window
def rates(data,window,step_size):
    rates=[]
    spikes=0
    time=0
    all_time=len(data)
    while time <= all_time-window:

        for i in range(time,window+time):
            #print(i)
            spikes+=data[i]

        rates.append(spikes/window)
        #rates.append(spikes/(window*10**-4)) #If I use the array of spikes/timestep
        time+=step_size
        #print(step)
        spikes=0
    return rates


# In[20]:


def rate_plots_myself(runtime,R_vip1rate,R_e23rate,R_pv23rate,R_sst23rate,R_vip23rate,
             R_e4rate,R_pv4rate,R_sst4rate,R_vip4rate,
             R_e5rate,R_pv5rate,R_sst5rate,R_vip5rate,
             R_e6rate,R_pv6rate,R_sst6rate,R_vip6rate,window,step):
    
#     window=200
#     step=1
    
    
    r_vip1= rates(R_vip1rate,window,step)

    r_e23=rates(R_e23rate,window,step)
    r_pv23=rates(R_pv23rate,window,step)
    r_sst23=rates(R_sst23rate,window,step)
    r_vip23=rates(R_vip23rate,window,step)

    r_e4=rates(R_e4rate,window,step)
    r_pv4=rates(R_pv4rate,window,step)
    r_sst4=rates(R_sst4rate,window,step)
    r_vip4=rates(R_vip4rate,window,step)

    r_e5=rates(R_e5rate,window,step)
    r_pv5=rates(R_pv5rate,window,step)
    r_sst5=rates(R_sst5rate,window,step)
    r_vip5=rates(R_vip5rate,window,step)

    r_e6=rates(R_e6rate,window,step)
    r_pv6=rates(R_pv6rate,window,step)
    r_sst6=rates(R_sst6rate,window,step)
    r_vip6=rates(R_vip6rate,window,step)

    #time=[i+window for i in range(0,len(r_pv5))]
    time=[i for i in range(0,len(r_pv5))]
    
    fig = plt.figure(figsize=(6,4))
    plt.plot(time,r_vip1,color='orange', label='vip')
    plt.xlabel('time (ms)')
    plt.ylabel('spikes/s')
    plt.legend()
    plt.title('Pops activity - layer 1')
    plt.show()


    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(15,9))
    f.suptitle('N=%sk, G=%s, B=%s'%(int(Ntot/1000),G,nu_ext),fontsize=10)
    ax1.plot(time, r_e23,color='r', label='e')
    ax1.plot(time, r_pv23,color='b', label='pv')
    ax1.plot(time, r_sst23,color='g', label='sst')
    ax1.plot(time, r_vip23,color='orange', label='vip')
#     ax1.xaxis.set_ticks(np.arange(0, (runtime/ms)*10+1, 1000)) # Ticks are placed at positions 0 to 8000 but the values displayed are from 0 to 800
#     ax1.xaxis.set_ticklabels(np.arange(0, int(runtime/ms)+1, 100))
    ax1.legend()
    ax1.set_xlabel('time (ms)')
    #ax1.set_xlim(300,800)
    #ax1.set_ylim(0,10)
    ax1.set_ylabel('spikes/s')
    ax1.set_title('Population rates - layer 2/3')

    ax2.plot(time, r_e4,color='r', label='e')
    ax2.plot(time, r_pv4,color='b', label='pv')
    ax2.plot(time, r_sst4,color='g', label='sst')
    ax2.plot(time, r_vip4,color='orange', label='vip')
#     ax2.xaxis.set_ticks(np.arange(0, (runtime/ms)*10+1, 1000)) # Ticks are placed at positions 0 to 8000 but the values displayed are from 0 to 800
#     ax2.xaxis.set_ticklabels(np.arange(0, int(runtime/ms)+1, 100))
    ax2.set_title('Population rates - layer 4')
    #ax2.set_xlim(300,800)
    #ax2.set_ylim(0,10)
    ax2.set_xlabel('time (ms)')
    ax2.set_ylabel('spikes/s')
    ax2.legend()

    ax3.plot(time, r_e5,color='r', label='e')
    ax3.plot(time, r_pv5,color='b', label='pv')
    ax3.plot(time, r_sst5,color='g', label='sst')
    ax3.plot(time, r_vip5,color='orange', label='vip')
#     ax3.xaxis.set_ticks(np.arange(0, (runtime/ms)*10+1, 1000)) # Ticks are placed at positions 0 to 8000 but the values displayed are from 0 to 800
#     ax3.xaxis.set_ticklabels(np.arange(0, int(runtime/ms)+1, 100))
    ax3.set_title('Population rates - layer 5')
    ax3.set_xlabel('time (ms)')
    #ax3.set_xlim(300,800)
    #ax3.set_ylim(0,20)
    ax3.set_ylabel('spikes/s')
    ax3.legend()

    ax4.plot(time, r_e6,color='r', label='e')
    ax4.plot(time, r_pv6,color='b', label='pv')
    ax4.plot(time, r_sst6,color='g', label='sst')
    ax4.plot(time, r_vip6,color='orange', label='vip')
#     ax4.xaxis.set_ticks(np.arange(0, (runtime/ms)*10+1, 1000)) # Ticks are placed at positions 0 to 8000 but the values displayed are from 0 to 800
#     ax4.xaxis.set_ticklabels(np.arange(0, int(runtime/ms)+1, 100))
    ax4.set_title('Population rates - layer 6' )
    ax4.set_xlabel('time (ms)')
    #ax4.set_xlim(300,800)
    #ax4.set_ylim(0,20)
    ax4.set_ylabel('spikes/s')
    ax4.legend()

    plt.subplots_adjust(left=0.125,
                        bottom=0.125, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.2, 
                        hspace=0.3)              

    plt.show()


# In[21]:


window=200
step=10
rate_plots_myself(runtime,R_vip1rate,R_e23rate,R_pv23rate,R_sst23rate,R_vip23rate,
             R_e4rate,R_pv4rate,R_sst4rate,R_vip4rate,
             R_e5rate,R_pv5rate,R_sst5rate,R_vip5rate,
             R_e6rate,R_pv6rate,R_sst6rate,R_vip6rate,window,step)


# In[22]:


# #TRY WITH ONE COMPUTATION OF RATE
# window=200
# step=10
# r_pv5=rates(R_pv5rate,window,step)
# #print(t)
# print(len(r_pv5))
# time=[i+window for i in range(0,len(r_pv5))]
# time=[i for i in range(0,len(r_pv5))]

# #print(time)


# r_vip1= rates(R_vip1rate,window,step)

# r_e23=rates(R_e23rate,window,step)
# r_pv23=rates(R_pv23rate,window,step)
# r_sst23=rates(R_sst23rate,window,step)
# r_vip23=rates(R_vip23rate,window,step)

# r_e4=rates(R_e4rate,window,step)
# r_pv4=rates(R_pv4rate,window,step)
# r_sst4=rates(R_sst4rate,window,step)
# r_vip4=rates(R_vip4rate,window,step)

# r_e5=rates(R_e5rate,window,step)
# r_pv5=rates(R_pv5rate,window,step)
# r_sst5=rates(R_sst5rate,window,step)
# r_vip5=rates(R_vip5rate,window,step)

# r_e6=rates(R_e6rate,window,step)
# r_pv6=rates(R_pv6rate,window,step)
# r_sst6=rates(R_sst6rate,window,step)
# r_vip6=rates(R_vip6rate,window,step)

#I chose which one to plot
# fig = plt.figure(figsize=(6,4))
# plt.plot(time,r_pv6,color='b', label='pv')
# plt.xlabel('time (ms)')
# plt.ylabel('sp/s')
# plt.legend()
# plt.title('Pops activity - layer 6')
# plt.show()


# In[23]:


#Import the tot number of spikes files
#layer1
S_vip1num_spikes=np.array(np.loadtxt('Spikes_files/S_vip1numspike.txt') )

#layer23
S_e23num_spikes=np.array(np.loadtxt('Spikes_files/S_e23numspike.txt') )
S_pv23num_spikes=np.array(np.loadtxt('Spikes_files/S_pv23numspike.txt') )
S_sst23num_spikes=np.array(np.loadtxt('Spikes_files/S_sst23numspike.txt') )
S_vip23num_spikes=np.array(np.loadtxt('Spikes_files/S_vip23numspike.txt') )

#layer4
S_e4num_spikes=np.array(np.loadtxt('Spikes_files/S_e4numspike.txt') )
S_pv4num_spikes=np.array(np.loadtxt('Spikes_files/S_pv4numspike.txt') )
S_sst4num_spikes=np.array(np.loadtxt('Spikes_files/S_sst4numspike.txt') )
S_vip4num_spikes=np.array(np.loadtxt('Spikes_files/S_vip4numspike.txt') )

#layer5
S_e5num_spikes=np.array(np.loadtxt('Spikes_files/S_e5numspike.txt') )
S_pv5num_spikes=np.array(np.loadtxt('Spikes_files/S_pv5numspike.txt') )
S_sst5num_spikes=np.array(np.loadtxt('Spikes_files/S_sst5numspike.txt') )
S_vip5num_spikes=np.array(np.loadtxt('Spikes_files/S_vip5numspike.txt') )

#layer6
S_e6num_spikes=np.array(np.loadtxt('Spikes_files/S_e6numspike.txt') )
S_pv6num_spikes=np.array(np.loadtxt('Spikes_files/S_pv6numspike.txt') )
S_sst6num_spikes=np.array(np.loadtxt('Spikes_files/S_sst6numspike.txt') )
S_vip6num_spikes=np.array(np.loadtxt('Spikes_files/S_vip6numspike.txt') ) 


# In[24]:


#------------------------------------------------------------------------------
# Compute the rates for all the neuron groups (if I record from all)
#------------------------------------------------------------------------------
def compute_FR(N,N1,runtime,S_vip1num_spikes,
              S_e4num_spikes,S_pv4num_spikes,S_sst4num_spikes,S_vip4num_spikes,
              S_e5num_spikes,S_pv5num_spikes,S_sst5num_spikes,S_vip5num_spikes,
              S_e6num_spikes,S_pv6num_spikes,S_sst6num_spikes,S_vip6num_spikes,
             S_e23num_spikes,S_pv23num_spikes,S_sst23num_spikes,S_vip23num_spikes):

    #runtime must be in seconds (it is in ms now)
    runtime=runtime/1000
    tot_sp_vip1 =  S_vip1num_spikes  #Layer 2/3
    rate_vip1= tot_sp_vip1/(N1*runtime)


    tot_sp_e23 =  S_e23num_spikes  #Layer 2/3
    #print(tot_sp_e23)
    #print(N[0][0])
    # Total number of spikes and rate for each group:
    rate_e23= tot_sp_e23/(N[0][0]*runtime)
    tot_sp_pv23 =  S_pv23num_spikes
    rate_pv23= tot_sp_pv23/(N[0][1]*runtime)
    tot_sp_sst23 =  S_sst23num_spikes
    rate_sst23= tot_sp_sst23/(N[0][2]*runtime)
    tot_sp_vip23 =  S_vip23num_spikes
    rate_vip23= tot_sp_vip23/(N[0][3]*runtime)


    tot_sp_e4 =  S_e4num_spikes
    rate_e4= tot_sp_e4/(N[1][0]*runtime)
    tot_sp_pv4 =  S_pv4num_spikes
    rate_pv4= tot_sp_pv4/(N[1][1]*runtime)
    tot_sp_sst4 =  S_sst4num_spikes
    rate_sst4= tot_sp_sst4/(N[1][2]*runtime)
    tot_sp_vip4 =  S_vip4num_spikes
    rate_vip4= tot_sp_vip4/(N[1][3]*runtime)

    tot_sp_e5 =  S_e5num_spikes
    rate_e5= tot_sp_e5/(N[2][0]*runtime)
    tot_sp_pv5 =  S_pv5num_spikes
    rate_pv5= tot_sp_pv5/(N[2][1]*runtime)
    tot_sp_sst5 =  S_sst5num_spikes
    rate_sst5= tot_sp_sst5/(N[2][2]*runtime)
    tot_sp_vip5 =  S_vip5num_spikes
    rate_vip5= tot_sp_vip5/(N[2][3]*runtime)


    tot_sp_e6 =  S_e6num_spikes
    rate_e6= tot_sp_e6/(N[3][0]*runtime)
    tot_sp_pv6 =  S_pv6num_spikes
    rate_pv6= tot_sp_pv6/(N[3][1]*runtime)
    tot_sp_sst6 =  S_sst6num_spikes
    rate_sst6= tot_sp_sst6/(N[3][2]*runtime)
    tot_sp_vip6 =  S_vip6num_spikes
    rate_vip6= tot_sp_vip6/(N[3][3]*runtime)
    
    print("-----------------------Computing the firing rates--------------------------------")
    print('rate_vip1: %f'%(rate_vip1))
    print('rate_e23: %f rate_pv23: %f rate_sst23: %f rate_vip23: %f'%(rate_e23,rate_pv23,rate_sst23,rate_vip23))
    print('rate_e4: %f rate_pv4: %f rate_sst4: %f rate_vip4: %f'%(rate_e4,rate_pv4,rate_sst4,rate_vip4))
    print('rate_e5: %f rate_pv5: %f rate_sst5: %f rate_vip5: %f'%(rate_e5,rate_pv5,rate_sst5,rate_vip5))
    print('rate_e6: %f rate_pv6: %f rate_sst6: %f rate_vip6: %f'%(rate_e6,rate_pv6,rate_sst6,rate_vip6))
    
    return rate_e23,rate_pv23,rate_sst23,rate_vip23,rate_e4,rate_pv4,rate_sst4,rate_vip4,rate_e5,rate_pv5,rate_sst5,rate_vip5,rate_e6,rate_pv6,rate_sst6,rate_vip6


# In[25]:


rate_e23,rate_pv23,rate_sst23,rate_vip23,rate_e4,rate_pv4,rate_sst4,rate_vip4,rate_e5,rate_pv5,rate_sst5,rate_vip5,rate_e6,rate_pv6,rate_sst6,rate_vip6=compute_FR(N,N1,runtime,S_vip1num_spikes,
              S_e4num_spikes,S_pv4num_spikes,S_sst4num_spikes,S_vip4num_spikes,
              S_e5num_spikes,S_pv5num_spikes,S_sst5num_spikes,S_vip5num_spikes,
              S_e6num_spikes,S_pv6num_spikes,S_sst6num_spikes,S_vip6num_spikes,
             S_e23num_spikes,S_pv23num_spikes,S_sst23num_spikes,S_vip23num_spikes)


# In[26]:


#------------------------------------------------------------------------------
# Compute the rates for all the neuron groups (if I record from a subfraction)
#------------------------------------------------------------------------------
def compute_FR_sub(n_activity,runtime,S_vip1num_spikes,
              S_e4num_spikes,S_pv4num_spikes,S_sst4num_spikes,S_vip4num_spikes,
              S_e5num_spikes,S_pv5num_spikes,S_sst5num_spikes,S_vip5num_spikes,
              S_e6num_spikes,S_pv6num_spikes,S_sst6num_spikes,S_vip6num_spikes,
             S_e23num_spikes,S_pv23num_spikes,S_sst23num_spikes,S_vip23num_spikes):

    #runtime must be in seconds (it is in ms now)
    runtime=runtime/1000
    tot_sp_vip1 =  S_vip1num_spikes  #Layer 2/3
    rate_vip1= tot_sp_vip1/(n_activity*runtime)


    tot_sp_e23 =  S_e23num_spikes  #Layer 2/3
    #print(tot_sp_e23)
    #print(N[0][0])
    # Total number of spikes and rate for each group:
    rate_e23= tot_sp_e23/(n_activity*runtime)
    tot_sp_pv23 =  S_pv23num_spikes
    rate_pv23= tot_sp_pv23/(n_activity*runtime)
    tot_sp_sst23 =  S_sst23num_spikes
    rate_sst23= tot_sp_sst23/(n_activity*runtime)
    tot_sp_vip23 =  S_vip23num_spikes
    rate_vip23= tot_sp_vip23/(n_activity*runtime)


    tot_sp_e4 =  S_e4num_spikes
    rate_e4= tot_sp_e4/(n_activity*runtime)
    tot_sp_pv4 =  S_pv4num_spikes
    rate_pv4= tot_sp_pv4/(n_activity*runtime)
    tot_sp_sst4 =  S_sst4num_spikes
    rate_sst4= tot_sp_sst4/(n_activity*runtime)
    tot_sp_vip4 =  S_vip4num_spikes
    rate_vip4= tot_sp_vip4/(n_activity*runtime)

    tot_sp_e5 =  S_e5num_spikes
    rate_e5= tot_sp_e5/(n_activity*runtime)
    tot_sp_pv5 =  S_pv5num_spikes
    rate_pv5= tot_sp_pv5/(n_activity*runtime)
    tot_sp_sst5 =  S_sst5num_spikes
    rate_sst5= tot_sp_sst5/(n_activity*runtime)
    tot_sp_vip5 =  S_vip5num_spikes
    rate_vip5= tot_sp_vip5/(n_activity*runtime)


    tot_sp_e6 =  S_e6num_spikes
    rate_e6= tot_sp_e6/(n_activity*runtime)
    tot_sp_pv6 =  S_pv6num_spikes
    rate_pv6= tot_sp_pv6/(n_activity*runtime)
    tot_sp_sst6 =  S_sst6num_spikes
    rate_sst6= tot_sp_sst6/(n_activity*runtime)
    tot_sp_vip6 =  S_vip6num_spikes
    rate_vip6= tot_sp_vip6/(n_activity*runtime)
    
    print("-----------------------Computing the firing rates--------------------------------")
    print('rate_vip1: %f'%(rate_vip1))
    print('rate_e23: %f rate_pv23: %f rate_sst23: %f rate_vip23: %f'%(rate_e23,rate_pv23,rate_sst23,rate_vip23))
    print('rate_e4: %f rate_pv4: %f rate_sst4: %f rate_vip4: %f'%(rate_e4,rate_pv4,rate_sst4,rate_vip4))
    print('rate_e5: %f rate_pv5: %f rate_sst5: %f rate_vip5: %f'%(rate_e5,rate_pv5,rate_sst5,rate_vip5))
    print('rate_e6: %f rate_pv6: %f rate_sst6: %f rate_vip6: %f'%(rate_e6,rate_pv6,rate_sst6,rate_vip6))
    
    return rate_e23,rate_pv23,rate_sst23,rate_vip23,rate_e4,rate_pv4,rate_sst4,rate_vip4,rate_e5,rate_pv5,rate_sst5,rate_vip5,rate_e6,rate_pv6,rate_sst6,rate_vip6


# In[44]:


#function to have only 2 decimals for FR
def decimals(value):
    formatted_string = "{:.2f}".format(value)
    float_value = float(formatted_string)
    return float_value


# In[45]:


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#plot bg data and data from the current simulation
def plot_meanFR(rate_e23,rate_pv23,rate_sst23,rate_vip23,rate_e4,rate_pv4,rate_sst4,rate_vip4,rate_e5,rate_pv5,rate_sst5,rate_vip5,rate_e6,rate_pv6,rate_sst6,rate_vip6
):

    labels = ['e','pv', 'sst', 'vip']


#     model_means23 =[0.30,2.59,3.34,6.99]
#     model_means4=[0.93,3.21,1.71,0.27]
#     model_means5=[1.95,4.02,3.60,5.36] #N=10K, 1 run of the program
#     model_means6=[0.87,3.55,5.86,2.75]
    
    model_means23 = [0.3,2.68,3.5,7.28]
    model_means4= [0.99,3.59,1.62,0.62] #N=5K, 10 run of the program, mean
    model_means5= [1.99,4.76,3.77,6.03]
    model_means6 = [0.96,4.28,6.08,2.84]
    
    errors_23 = [0.01,0.151715155587104,0.085742306601062,0.122794003635551]
    errors_4=[0.018839538991908,0.105932586407721,0.137976871453621,0.096935200953077]
    errors_5=[0.044113387640075,0.155145592054941,0.105516761857441,0.52503880650315]
    errors_6=[0.024620615854391,0.104608347188083,0.128567258402174,0.168722318392471]

    
    

    rate_e23f=decimals(rate_e23)
    rate_pv23f=decimals(rate_pv23)
    rate_sst23f=decimals(rate_sst23)
    rate_vip23f=decimals(rate_vip23)
    
    rate_e4f=decimals(rate_e4)
    rate_pv4f=decimals(rate_pv4)
    rate_sst4f=decimals(rate_sst4)
    rate_vip4f=decimals(rate_vip4)
    
    rate_e5f=decimals(rate_e5)
    rate_pv5f=decimals(rate_pv5)
    rate_sst5f=decimals(rate_sst5)
    rate_vip5f=decimals(rate_vip5)
    
    rate_e6f=decimals(rate_e6)
    rate_pv6f=decimals(rate_pv6)
    rate_sst6f=decimals(rate_sst6)
    rate_vip6f=decimals(rate_vip6)
    


    model_means23_I = [rate_e23f,rate_pv23f,rate_sst23f,rate_vip23f]
    model_means4_I = [rate_e4f,rate_pv4f,rate_sst4f,rate_vip4f]
    model_means5_I = [rate_e5f,rate_pv5f,rate_sst5f,rate_vip5f]
    model_means6_I = [rate_e6f,rate_pv6f,rate_sst6f,rate_vip6f]


    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars
    w=0.3
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(16,10))
    #fig, ax = plt.subplots(figsize=(10,5))
    f.suptitle('FR of pops MODEL - background noise vs Input',fontsize=15)

    rects1_23 = ax1.bar(x - width/2, model_means23,yerr=errors_23, width=w,color='b', label='bg')
    rects2_23 = ax1.bar(x + width/2, model_means23_I, width,color='r', label='bg+I')

    rects1_4 = ax2.bar(x - width/2, model_means4,yerr=errors_4, width=w,color='b', label='bg')
    rects2_4 = ax2.bar(x + width/2, model_means4_I, width,color='r',  label='bg+I')

    rects1_5 = ax3.bar(x - width/2, model_means5,yerr=errors_5,width=w,color='b', label='bg')
    rects2_5 = ax3.bar(x + width/2, model_means5_I, width,color='r',  label='bg+I')

    rects1_6 = ax4.bar(x - width/2, model_means6,yerr=errors_6, width=w,color='b', label='bg')
    rects2_6 = ax4.bar(x + width/2, model_means6_I, width,color='r',  label='bg+I')



    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax1.set_ylabel('Spikes/s', fontsize=15)
    ax1.set_title('Firing rates of pops L2/3',fontsize=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=15)
    ax1.legend()

    ax2.set_ylabel('Spikes/s', fontsize=15)
    ax2.set_title('Firing rates of pops L4',fontsize=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=15)
    ax2.legend()

    ax3.set_ylabel('Spikes/s', fontsize=15)
    ax3.set_title('Firing rates of pops L5',fontsize=15)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=15)
    ax3.legend()

    ax4.set_ylabel('Spikes/s', fontsize=15)
    ax4.set_title('Firing rates of pops L6',fontsize=15)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, fontsize=15)
    ax4.legend()

    def autolabel(rects,ax):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1_23,ax1)
    autolabel(rects2_23,ax1)


    autolabel(rects1_4,ax2)
    autolabel(rects2_4,ax2)

    autolabel(rects1_5,ax3)
    autolabel(rects2_5,ax3)

    autolabel(rects1_6,ax4)
    autolabel(rects2_6,ax4)


    f.tight_layout()

    plt.show()


# In[46]:


plot_meanFR(rate_e23,rate_pv23,rate_sst23,rate_vip23,rate_e4,rate_pv4,rate_sst4,rate_vip4,rate_e5,rate_pv5,rate_sst5,rate_vip5,rate_e6,rate_pv6,rate_sst6,rate_vip6
)


# In[30]:


#COMPUTED WITH 10 RUN PF THE PROGRAM. 5K mean values
rate_e23m= 0.301375404443
rate_pv23m= 2.679999999
rate_sst23m= 3.497872341
rate_vip23m= 7.283177569
rate_e4m= 0.985544554
rate_pv4m= 3.594557823
rate_sst4m= 1.62201258
rate_vip4m= 0.623456789
rate_e5m= 1.98731444
rate_pv5m= 4.76031746
rate_sst5m= 3.770238095
rate_vip5m= 6.03030303
rate_e6m= 0.955344419
rate_pv6m= 4.276143792
rate_sst6m= 6.077777778
rate_vip6m= 2.843859649


# In[31]:


#tau_NMDA= 80,the error COMPUTED WITH 10 RUN PF THE PROGRAM. 5K
#bg data from experiments compared with my simulation
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


rate_e23f=decimals(rate_e23m)
rate_pv23f=decimals(rate_pv23m)
rate_sst23f=decimals(rate_sst23m)
rate_vip23f=decimals(rate_vip23m)

rate_e4f=decimals(rate_e4m)
rate_pv4f=decimals(rate_pv4m)
rate_sst4f=decimals(rate_sst4m)
rate_vip4f=decimals(rate_vip4m)

rate_e5f=decimals(rate_e5m)
rate_pv5f=decimals(rate_pv5m)
rate_sst5f=decimals(rate_sst5m)
rate_vip5f=decimals(rate_vip5m)

rate_e6f=decimals(rate_e6m)
rate_pv6f=decimals(rate_pv6m)
rate_sst6f=decimals(rate_sst6m)
rate_vip6f=decimals(rate_vip6m)
    


model_means23 = [rate_e23f,rate_pv23f,rate_sst23f,rate_vip23f]
model_means4= [rate_e4f,rate_pv4f,rate_sst4f,rate_vip4f]
model_means5= [rate_e5f,rate_pv5f,rate_sst5f,rate_vip5f]
model_means6 = [rate_e6f,rate_pv6f,rate_sst6f,rate_vip6f]

errors_23 = [0.009802117298079,0.151715155587104,0.085742306601062,0.122794003635551]
errors_4=[0.018839538991908,0.105932586407721,0.137976871453621,0.096935200953077]
errors_5=[0.044113387640075,0.155145592054941,0.105516761857441,0.52503880650315]
errors_6=[0.024620615854391,0.104608347188083,0.128567258402174,0.168722318392471]


labels = ['e','pv', 'sst', 'vip']
exp_means23 = [0.27, 3.10, 3.63, 7.97]
exp_means4 = [1.06, 3.89, 1.89, 0.85]
exp_means5 = [2.22, 4.58, 3.78, 6.50]
exp_means6 = [0.94, 4.61, 6.30, 2.90]



x = np.arange(len(labels))  # the label locations
w = 0.3  # the width of the bars
width=0.3

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(16,10))
#fig, ax = plt.subplots(figsize=(10,5))
f.suptitle('FR of pops - background noise',fontsize=15)
rects1_23 = ax1.bar(x - width/2, exp_means23, width=w, label='experiments')
rects2_23 = ax1.bar(x + width/2, model_means23,yerr=errors_23, width=w, label='model')

rects1_4 = ax2.bar(x - width/2, exp_means4, width=w, label='experiments')
rects2_4 = ax2.bar(x + width/2, model_means4,yerr=errors_4, width=w, label='model')

rects1_5 = ax3.bar(x - width/2, exp_means5, width=w, label='experiments')
rects2_5 = ax3.bar(x + width/2, model_means5,yerr=errors_5, width=w, label='model')

rects1_6 = ax4.bar(x - width/2, exp_means6, width=w, label='experiments')
rects2_6 = ax4.bar(x + width/2, model_means6,yerr=errors_6, width=w, label='model')




# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_ylabel('Spikes/s', fontsize=15)
ax1.set_title('Firing rates of pops L2/3',fontsize=15)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=15)
ax1.legend()

ax2.set_ylabel('Spikes/s', fontsize=15)
ax2.set_title('Firing rates of pops L4',fontsize=15)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=15)
ax2.legend()

ax3.set_ylabel('Spikes/s', fontsize=15)
ax3.set_title('Firing rates of pops L5',fontsize=15)
ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=15)
ax3.legend()

ax4.set_ylabel('Spikes/s', fontsize=15)
ax4.set_title('Firing rates of pops L6',fontsize=15)
ax4.set_xticks(x)
ax4.set_xticklabels(labels, fontsize=15)
ax4.legend()

def autolabel(rects,ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')



autolabel(rects1_23,ax1)
autolabel(rects2_23,ax1)


autolabel(rects1_4,ax2)
autolabel(rects2_4,ax2)

autolabel(rects1_5,ax3)
autolabel(rects2_5,ax3)

autolabel(rects1_6,ax4)
autolabel(rects2_6,ax4)


f.tight_layout()

plt.show()


# In[32]:


#Plots of connections


# In[33]:


Cp = np.loadtxt('connectionsPro.txt')
Cs=np.loadtxt('connectionsStren.txt')


# In[34]:


targets=[[0,0],[0,1],[0,2],[0,3],
        [1,0],[1,1],[1,2],[1,3],
        [2,0],[2,1],[2,2],[2,3],
        [3,0],[3,1],[3,2],[3,3]]
sources=[[0,0],[0,1],[0,2],[0,3],
        [1,0],[1,1],[1,2],[1,3],
        [2,0],[2,1],[2,2],[2,3],
        [3,0],[3,1],[3,2],[3,3]]

n_co= [[0 for x in range(0,len(sources))] for y in range(0,len(targets))] #create the GH matrix filled with 0 
#print(n_co)
n_out_co= [[0 for x in range(0,len(sources))] for y in range(0,len(targets))] #create the GH matrix filled with 0 

n_co_log= [[0 for x in range(0,len(sources))] for y in range(0,len(targets))] #create the GH matrix filled with 0 

strengh_co= [[0 for x in range(0,len(sources))] for y in range(0,len(targets))] #create the GH matrix filled with 0 

measure_co= [[0 for x in range(0,len(sources))] for y in range(0,len(targets))] #create the GH matrix filled with 0 


Cp_vis= [[0 for x in range(0,len(sources))] for y in range(0,len(targets))]
Cs_vis=[[0 for x in range(0,len(sources))] for y in range(0,len(targets))]

for h in range(len(sources)):
    for k in range(len(targets)):
        s_layer = sources[h][0]
        s_cell_type = sources[h][1]
        t_layer = targets[k][0]
        t_cell_type = targets[k][1]
        n_conn=Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*N[s_layer][s_cell_type]*N[t_layer][t_cell_type]
        n_out=Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*N[t_layer][t_cell_type]

        n_co_log[h][k]=np.log(1+n_conn)
        n_co[h][k]=n_conn
        n_out_co[h][k]=n_out

        
        if Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]==0:
            streng_conn=0
        else:
            streng_conn=G*Cs[4*s_layer+s_cell_type][4*t_layer+t_cell_type]/(Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*N[s_layer][s_cell_type])
    
        strengh_co[h][k]=streng_conn
        Cp_vis[h][k]=Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]
        Cs_vis[h][k]=Cs[4*s_layer+s_cell_type][4*t_layer+t_cell_type]

#print(n_connections)
#resh=np.reshape(n_connections, (16,16))
#print(resh)
#print(n_co)
#measure of connection between pops (number*streght)
for h in range(0,16):
    for k in range(0,16):
        measure_co[h][k]= n_co[h][k]*strengh_co[h][k]

        
allen_mes= np.multiply(Cs_vis,Cp_vis)
#measure_log= np.multiply(n_co_log,strengh_co) #no sense up to me
measure_out= np.multiply(n_out_co,strengh_co)




import seaborn as sns
from matplotlib.colors import LogNorm, Normalize
fig = plt.figure(figsize=(15, 10)) #create the figure
x_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis
y_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis

conn_plot = sns.heatmap(np.array(n_co),square=True,norm=LogNorm(),cmap='Reds',xticklabels=x_axis_labels, yticklabels=y_axis_labels) # plot GH with heatmap
conn_plot.set_xlabel('Targets', fontsize=15)
conn_plot.set_ylabel('Sources', fontsize=15)
plt.title('Connection numbers',fontsize=18)
plt.show(conn_plot)


fig = plt.figure(figsize=(15, 10)) #create the figure
x_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis
y_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis

coout_plot = sns.heatmap(np.array(n_out_co),square=True,norm=LogNorm(),cmap='Reds',xticklabels=x_axis_labels, yticklabels=y_axis_labels) # plot GH with heatmap
coout_plot.set_xlabel('Targets', fontsize=15)
coout_plot.set_ylabel('Sources', fontsize=15)
plt.title('Connection numbers- outgoing conn per neuron',fontsize=18)
plt.show(coout_plot)



#print(n_co)

fig = plt.figure(figsize=(15, 10)) #create the figure
x_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis
y_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis

cp_plot = sns.heatmap(np.array(Cp_vis),square=True,norm=LogNorm(),cmap='Reds',xticklabels=x_axis_labels, yticklabels=y_axis_labels) # plot GH with heatmap
cp_plot.set_xlabel('Targets', fontsize=15)
cp_plot.set_ylabel('Sources', fontsize=15)
plt.title('Connection prob Allen',fontsize=18)
plt.show(cp_plot)




fig = plt.figure(figsize=(15, 10)) #create the figure
x_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis
y_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis

strenght_plot = sns.heatmap(np.array(strengh_co),square=True,norm=LogNorm(),cmap='Reds',xticklabels=x_axis_labels, yticklabels=y_axis_labels) # plot GH with heatmap
strenght_plot.set_xlabel('Targets', fontsize=15)
strenght_plot.set_ylabel('Sources', fontsize=15)
plt.title('Conn strenghts',fontsize=18)
plt.show(strenght_plot)


fig = plt.figure(figsize=(15, 10)) #create the figure
x_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis
y_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis

cs_plot = sns.heatmap(np.array(Cs_vis),square=True,norm=LogNorm(),cmap='Reds',xticklabels=x_axis_labels, yticklabels=y_axis_labels) # plot GH with heatmap
cs_plot.set_xlabel('Targets', fontsize=15)
cs_plot.set_ylabel('Sources', fontsize=15)
plt.title('Connection strengh Allen',fontsize=18)
plt.show(cs_plot)




fig = plt.figure(figsize=(15, 10)) #create the figure
x_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis
y_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis

meas_plot = sns.heatmap(np.array(measure_co),square=True,norm=LogNorm(),cmap='Reds',xticklabels=x_axis_labels, yticklabels=y_axis_labels) # plot GH with heatmap
meas_plot.set_xlabel('Targets', fontsize=15)
meas_plot.set_ylabel('Sources', fontsize=15)
plt.title('Measure conn',fontsize=18)
plt.show(meas_plot)


fig = plt.figure(figsize=(15, 10)) #create the figure
x_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis
y_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis

mout_plot = sns.heatmap(np.array(measure_out),square=True,norm=LogNorm(),cmap='Reds',xticklabels=x_axis_labels, yticklabels=y_axis_labels) # plot GH with heatmap
mout_plot.set_xlabel('Targets', fontsize=15)
mout_plot.set_ylabel('Sources', fontsize=15)
plt.title('Measure conn/sending pop',fontsize=18)
plt.show(mout_plot)


fig = plt.figure(figsize=(15, 10)) #create the figure
x_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis
y_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis

#log no sense the way is computed!
# m_plot = sns.heatmap(measure_log,square=True,cmap='hot',xticklabels=x_axis_labels, yticklabels=y_axis_labels) # plot GH with heatmap
# m_plot.set_xlabel('Targets', fontsize=15)
# m_plot.set_ylabel('Sources', fontsize=15)
# plt.title('Measure conn',fontsize=18)
# plt.show(m_plot)


fig = plt.figure(figsize=(15, 10)) #create the figure
x_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis
y_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis

mA_plot = sns.heatmap(np.array(allen_mes)+1,square=True,norm=LogNorm(),cmap='hot',xticklabels=x_axis_labels, yticklabels=y_axis_labels) # plot GH with heatmap
mA_plot.set_xlabel('Targets', fontsize=15)
mA_plot.set_ylabel('Sources', fontsize=15)
plt.title('Measure conn ALLEN',fontsize=18)
plt.show(mA_plot)


# In[35]:


# fig = plt.figure(figsize=(15, 10)) #create the figure
# x_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis
# y_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis

# conn_plot = sns.heatmap(n_co,square=True,cmap='hot',xticklabels=x_axis_labels, yticklabels=y_axis_labels) # plot GH with heatmap
# conn_plot.set_xlabel('Targets', fontsize=15)
# conn_plot.set_ylabel('Sources', fontsize=15)
# plt.title('Connection numbers',fontsize=18)
# plt.show(conn_plot)

# fig = plt.figure(figsize=(15, 10)) #create the figure
# x_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis
# y_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis

# conn_plot = sns.heatmap(n_co_log,square=True,cmap='hot',xticklabels=x_axis_labels, yticklabels=y_axis_labels) # plot GH with heatmap
# conn_plot.set_xlabel('Targets', fontsize=15)
# conn_plot.set_ylabel('Sources', fontsize=15)
# plt.title('Connection numbers log',fontsize=18)
# plt.show(conn_plot)


# fig = plt.figure(figsize=(15, 10)) #create the figure
# x_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis
# y_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis

# coout_plot = sns.heatmap(n_out_co,square=True,cmap='hot',xticklabels=x_axis_labels, yticklabels=y_axis_labels) # plot GH with heatmap
# coout_plot.set_xlabel('Targets', fontsize=15)
# coout_plot.set_ylabel('Sources', fontsize=15)
# plt.title('Connection numbers- outgoing conn per neuron',fontsize=18)
# plt.show(coout_plot)



# #print(n_co)

# fig = plt.figure(figsize=(15, 10)) #create the figure
# x_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis
# y_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis

# cp_plot = sns.heatmap(Cp_vis,square=True,cmap='hot',xticklabels=x_axis_labels, yticklabels=y_axis_labels) # plot GH with heatmap
# cp_plot.set_xlabel('Targets', fontsize=15)
# cp_plot.set_ylabel('Sources', fontsize=15)
# plt.title('Connection prob Allen',fontsize=18)
# plt.show(cp_plot)




# fig = plt.figure(figsize=(15, 10)) #create the figure
# x_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis
# y_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis

# strenght_plot = sns.heatmap(strengh_co,square=True,cmap='hot',xticklabels=x_axis_labels, yticklabels=y_axis_labels) # plot GH with heatmap
# strenght_plot.set_xlabel('Targets', fontsize=15)
# strenght_plot.set_ylabel('Sources', fontsize=15)
# plt.title('Conn strenghts',fontsize=18)
# plt.show(strenght_plot)


# fig = plt.figure(figsize=(15, 10)) #create the figure
# x_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis
# y_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis

# cs_plot = sns.heatmap(Cs_vis,square=True,cmap='hot',xticklabels=x_axis_labels, yticklabels=y_axis_labels) # plot GH with heatmap
# cs_plot.set_xlabel('Targets', fontsize=15)
# cs_plot.set_ylabel('Sources', fontsize=15)
# plt.title('Connection strengh Allen',fontsize=18)
# plt.show(cs_plot)




# fig = plt.figure(figsize=(15, 10)) #create the figure
# x_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis
# y_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis

# meas_plot = sns.heatmap(measure_co,square=True,cmap='hot',xticklabels=x_axis_labels, yticklabels=y_axis_labels) # plot GH with heatmap
# meas_plot.set_xlabel('Targets', fontsize=15)
# meas_plot.set_ylabel('Sources', fontsize=15)
# plt.title('Measure conn',fontsize=18)
# plt.show(meas_plot)


# fig = plt.figure(figsize=(15, 10)) #create the figure
# x_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis
# y_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis

# mout_plot = sns.heatmap(measure_out,square=True,cmap='hot',xticklabels=x_axis_labels, yticklabels=y_axis_labels) # plot GH with heatmap
# mout_plot.set_xlabel('Targets', fontsize=15)
# mout_plot.set_ylabel('Sources', fontsize=15)
# plt.title('Measure conn/sending pop',fontsize=18)
# plt.show(mout_plot)


# fig = plt.figure(figsize=(15, 10)) #create the figure
# x_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis
# y_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis

# #log no sense the way is computed!
# # m_plot = sns.heatmap(measure_log,square=True,cmap='hot',xticklabels=x_axis_labels, yticklabels=y_axis_labels) # plot GH with heatmap
# # m_plot.set_xlabel('Targets', fontsize=15)
# # m_plot.set_ylabel('Sources', fontsize=15)
# # plt.title('Measure conn',fontsize=18)
# # plt.show(m_plot)


# fig = plt.figure(figsize=(15, 10)) #create the figure
# x_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis
# y_axis_labels = ['e23','pv23','sst23','vip23','e4','pv4','sst4','vip4','e5','pv5','sst5','vip5','e6','pv6','sst6','vip6'] # labels for x-axis

# mA_plot = sns.heatmap(allen_mes,square=True,cmap='hot',xticklabels=x_axis_labels, yticklabels=y_axis_labels) # plot GH with heatmap
# mA_plot.set_xlabel('Targets', fontsize=15)
# mA_plot.set_ylabel('Sources', fontsize=15)
# plt.title('Measure conn ALLEN',fontsize=18)
# plt.show(mA_plot)

