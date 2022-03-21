#!/usr/bin/env python
# coding: utf-8

# In[1]:


#final version
#This program simulate my netowork and save in files all the date produced by it.


# In[2]:


from brian2 import *
import numpy as np
import time
import matplotlib.pyplot as plt
from brian2tools import *
from brian2 import profiling_summary
import statistics


# In[3]:


#clear_cache('cython')


# In[4]:


startbuild = time.time()
print("Initializing and building the Network")


# In[5]:


######Parameters to change in the simulation######
runtime = 3000.0 * ms 
dt_sim=0.1 #ms
G=5 #global constant to increase weight 
Gl1=5
Ntot=5000 #Ntot=226562 Allen inst

#Percentage of AMPA and NMDA
e_ampa=0.8
e_nmda=0.2
i_ampa=0.8
i_nmda=0.2


#external input
#Iext=np.loadtxt('Iext0.txt')
Iext_l1= 0 
Iext=np.loadtxt('Iext.txt')
#Iext1=np.loadtxt('Iext1.txt')

#Background noise
nu_ext=np.loadtxt('nu_ext_try_g=5_new.txt')
nu_extl1= 650 *Hz

#IF I WANT DIFFERNT weight for each group for the external background noise 
#wext=np.loadtxt('wext_old.txt')


# In[6]:


N1=int(0.0192574218*Ntot) # not included in the calculation of Ntot, Ntot is just the 4 layers. 
N2_3=int(0.291088453*Ntot) # This percentage are computed looking at the numbers of the Allen institute 
N4=int(0.237625904*Ntot)
N5=int(0.17425693*Ntot)
N6= Ntot-N2_3-N4-N5
#N6=int(0.297031276*Ntot)
#print(N2_3+N4+N5+N6)

print("The corticular column in this model is composed by 4 layers+1")
print("Total number of neurons in the column: %s + %s \n85 perc excitatory and 15perc inhibitory \nIn each layer: 1 excitotory population and 3 inhibitory populations: pv, sst and vip cells.   "%(Ntot,N1))


perc_tot=np.loadtxt('perc_tot.txt') #Matrix containing the percentage of excitatory and inhib neurons
#print(perc_tot)
perc=np.loadtxt('perc.txt') #Matrix containing the percentage of neurons for each type in each layer
#print(perc)
n_tot= np.array([[N2_3,N2_3,N2_3,N2_3],[N4,N4,N4,N4],[N5,N5,N5,N5],[N6,N6,N6,N6]]) # number of total neuron in each layer
#print(n_tot)

N=perc*perc_tot*n_tot
N=np.matrix.round(N)
N=N.astype(int)
for k in range(0,4):
    N[k][0]+= n_tot[k][0]-sum(N[k])
print("Ntot of 4layers:")
print(N)
print("+ in layer 1 we have:%s neurons"%N1)
print(sum(N))

print('--------------------------------------------------')


# In[7]:


# #FOR THE MOMENT I AM NOT USING DISTINCTION BETWEEN E and I using the following!
# # Connectivity - external connections
# g_AMPA_ext_E = 1.0 * nS# 2.08 * nS #2.1 * nS 
# g_AMPA_ext_I = 1.0 * nS#1.62 * nS
# #ampa connections
# g_AMPA_rec_I =1.0 * nS #0.081 * nS # gEI_AMPA = 0.04 * nS    # Weight of excitatory to inhibitory synapses (AMPA)
# g_AMPA_rec_E =1.0 * nS# 0.104 * nS # gEE_AMPA = 0.05 * nS    # Weight of AMPA synapses between excitatory neurons

# # NMDA (excitatory)
# g_NMDA_E =1.0 * nS# 0.327 * nS #0.165 * nS # Weight of NMDA synapses between excitatory
# g_NMDA_I =1.0 * nS #0.258 * nS #0.13 * nS 

# # GABAergic (inhibitory)
# g_GABA_E =1.0 * nS #1.25 * nS # gIE_GABA = 1.3 * nS # Weight of inhibitory to excitatory synapses
# g_GABA_I =1.0 * nS# 0.973 * nS # gII_GABA = 1.0 * nS # Weight of inhibitory to inhibitory synapses


# In[8]:


# Synapse model
w_ext=1                                   #weight for each group for the external background noise, SAME fro everyone
V_E = 0 * mV                               # Reversal potential of excitatory synapses
#V_I = -70 * mV  each group has his own!!    # Reversal potential of inhibitory synapses
tau_AMPA = 2.0 * ms                          # Decay constant of AMPA-type conductances
tau_GABA = 5.0 * ms                          # Decay constant of GABA-type conductances
tau_NMDA_decay = 80.0 * ms                  # Decay constant of NMDA-type conductances
tau_NMDA_rise = 2.0 * ms                     # Rise constant of NMDA-type conductances
alpha_NMDA = 0.5 * kHz                       # Saturation constant of NMDA-type conductances
Mg2 = 1.
d = 2 * ms                                 # Transmission delay of recurrent excitatory and inhibitory connections


# In[9]:


print("Importing the data")

#Matrix containing all the connections (probabilities and strenght)
Cp = np.loadtxt('connectionsPro.txt')
Cs=np.loadtxt('connectionsStren.txt')

Cpl1 = np.loadtxt('Cpl1.txt')
Csl1=np.loadtxt('Csl1.txt')
Cp_tol1 = np.loadtxt('Cptol1.txt')
Cs_tol1 = np.loadtxt('Cstol1.txt')
#print(Cp_tol1)
Cs_l1_l1=1.73
Cp_l1_l1=0.656
# print(Cs)
# Cs[4*1+0][4*1+1]


#Parameters of the neurons for each layer 
# row: layer in this order from top to bottom: 2_3,4,5,6
# column: populations in this order: e, pv, sst, vip
Cm=np.loadtxt('Cm.txt') #pF
gl=np.loadtxt('gl.txt') #nS
Vl=np.loadtxt('Vl.txt') #mV
Vr=np.loadtxt('Vr.txt') #mV
Vt=np.loadtxt('Vt.txt') #mV
tau_ref=np.loadtxt('tau_ref.txt') #ms


#Parameters of layer 1
Nl1= N1
Vt_l1= -40.20
Vr_l1= -65.5
Cm_l1= 37.11
gl_l1= 4.07
Vl_l1= -65.5
tau_ref_l1= 3.5

########Check if everything is correct##########
#print('------------------Check--------------------------------')
# print('Cm')
# print(Cm)
# print('gl')
# print(gl)
#print(Vl)
#print(Vt)
#print(type(Vt[0][1]))
# print(tau_ref)

#Comuting tau for all the neurons
# tau=Cm*1./gl
# print('tau')
# print(tau)

#I can use this to change tau
#gl[1][0]=1.3
#Cm[1][3]=100
#Cm[3][1]=200
#gl[3][1]=1.5
#print(Cm)
# for r in range(0,4):
#     gl[r][2]=40
# print('gl')
# print(gl)

#Comuting tau for all the neurons AFTER the change
# tau=Cm*1./gl
# print('tau new')
# print(tau)
# print('gl new')
# print(gl)
print('--------------------------------------------------')


# In[10]:


taupre = 20*ms
taupost = 20*ms
wmax = 20
dApre = 0.01
dApost = -dApre*taupre/taupost*1.05


# In[11]:


#times for the inputs
t1=500
t2=1100
t3=2100
t4=2300


# In[12]:


#Equations of the model
eqs='''
        dv / dt = (- g_m * (v - V_L) - I_syn) / C_m : volt (unless refractory)
        I_syn = I_AMPA_rec + I_AMPA_ext + I_GABA + I_NMDA + I_external: amp

        C_m : farad
        g_m: siemens
        V_L : volt
        g_AMPA_ext: siemens
        g_AMPA_rec : siemens
        g_NMDA : siemens
        g_GABA :siemens
        
        #If I want same input for the entire simulation
        #I_external = I_ext: amp 
        
        #When I want no input and then input activated
        I_external= (abs(t-t1*ms)/(t-t1*ms) + 1)* (I_ext/2) : amp #at the beginnig is 0 then the input is activated
        
        #If I want: at the beginnig I is 0 then the input is activated then deactivated then activated again
        #I_external= (abs(t-t1*ms)/(t-t1*ms) + 1) * (I_ext/2)-(abs(t-t2*ms)/(t-t2*ms) + 1) * (I_ext/2) + (abs(t-t3*ms)/(t-t3*ms) + 1) * (I_ext/2)- (abs(t-t4*ms)/(t-t4*ms) + 1) * (I_ext/2) : amp 
        
        #When I have 2 inputs at different times going to different layers
        #I_external= (abs(t-t1*ms)/(t-t1*ms) + 1)* (I_ext/2) + (abs(t-t2*ms)/(t-t2*ms) + 1)* (I_ext1/2) : amp #at the beginnig is 0 then the input is activated
        #I_ext1 : amp #the second input to the other layer
        
        I_ext : amp
        
        
        
        
        I_AMPA_ext= g_AMPA_ext * (v - V_E) * w_ext * s_AMPA_ext : amp
        ds_AMPA_ext / dt = - s_AMPA_ext / tau_AMPA : 1
        #w_ext: 1 (If you want to have different weight fo each group, use this and uncomment later in pop.w_ext)
        # Here I don't need the summed variable because the neuron receive inputs from only one Poisson generator. 
        #Each neuron need only one s.


        I_AMPA_rec = g_AMPA_rec * (v - V_E) * 1 * s_AMPA_tot : amp
        s_AMPA_tot=s_AMPA_tot0+s_AMPA_tot1+s_AMPA_tot2+s_AMPA_tot3 : 1
        s_AMPA_tot0 : 1
        s_AMPA_tot1 : 1
        s_AMPA_tot2 : 1
        s_AMPA_tot3 : 1
        #the eqs_ampa solve many s and sum them and give the summed value here
        #Each neuron receives inputs from many neurons. Each of them has his own differential equation s_AMPA (where I have the deltas with the spikes). 
        #I then sum all the solutions s of the differential equations and I obtain s_AMPA_tot_post.
        #One s_AMPA_tot from each group of neurons sending excitatotion (each neruon is receiving from 4 groups)

        I_GABA= g_GABA * (v - V_I) * s_GABA_tot : amp
        V_I : volt
        
        s_GABA_tot=s_GABA_tot0+s_GABA_tot1+s_GABA_tot2+s_GABA_tot3+s_GABA_tot4+s_GABA_tot5
                    +s_GABA_tot6+s_GABA_tot7+s_GABA_tot8+s_GABA_tot9+s_GABA_tot10+s_GABA_tot11+s_GABA_tot12: 1
        s_GABA_tot0 : 1
        s_GABA_tot1 : 1
        s_GABA_tot2 : 1
        s_GABA_tot3 : 1
        s_GABA_tot4 : 1
        s_GABA_tot5 : 1
        s_GABA_tot6 : 1
        s_GABA_tot7 : 1
        s_GABA_tot8 : 1
        s_GABA_tot9 : 1
        s_GABA_tot10 : 1
        s_GABA_tot11 : 1
        s_GABA_tot12: 1 
        



        I_NMDA  = g_NMDA * (v - V_E) / (1 + Mg2 * exp(-0.062 * v / mV) / 3.57) * s_NMDA_tot : amp
        s_NMDA_tot=s_NMDA_tot0+s_NMDA_tot1+s_NMDA_tot2+s_NMDA_tot3 : 1
        s_NMDA_tot0 : 1
        s_NMDA_tot1 : 1
        s_NMDA_tot2 : 1
        s_NMDA_tot3 : 1

     '''
    
eqs_ampa_base='''
            s_AMPA_tot_post= w_AMPA* s_AMPA : 1 (summed)  
            ds_AMPA / dt = - s_AMPA / tau_AMPA : 1 (clock-driven)
            w_AMPA: 1
            dapre/dt = -apre/taupre : 1 (event-driven)  #(clock-driven) 
            dapost/dt = -apost/taupost : 1 (event-driven)  
        '''
        
eqs_ampa=[]

for k in range (4):
    eqs_ampa.append(eqs_ampa_base.replace('s_AMPA_tot_post','s_AMPA_tot'+str(k)+'_post'))


eqs_nmda_base='''s_NMDA_tot_post = w_NMDA * s_NMDA : 1 (summed)
    ds_NMDA / dt = - s_NMDA / tau_NMDA_decay + alpha_NMDA * x * (1 - s_NMDA) : 1 (clock-driven)
    dx / dt = - x / tau_NMDA_rise : 1 (clock-driven)
    w_NMDA : 1            
    dapre/dt = -apre/taupre : 1 (event-driven)  #(clock-driven) 
    dapost/dt = -apost/taupost : 1 (event-driven)  
'''
eqs_nmda=[]
for k in range (4):
    eqs_nmda.append(eqs_nmda_base.replace('s_NMDA_tot_post','s_NMDA_tot'+str(k)+'_post'))

eqs_gaba_base='''
    s_GABA_tot_post= w_GABA* s_GABA : 1 (summed)  
    ds_GABA/ dt = - s_GABA/ tau_GABA : 1 (clock-driven)
    w_GABA: 1  
'''    

eqs_gaba=[]
for k in range (12):
    eqs_gaba.append(eqs_gaba_base.replace('s_GABA_tot_post','s_GABA_tot'+str(k)+'_post'))
    
#Eqs I need to use for L1 connections    
eqs_gaba_l1= '''s_GABA_tot12_post= w_GABA* s_GABA : 1 (summed)  
    ds_GABA/ dt = - s_GABA/ tau_GABA : 1 (clock-driven)
    w_GABA: 1
'''   


# In[13]:


end_import= time.time()


# In[14]:


start_populations= time.time()
#def create_populations(N,eqs,Vt,Vr,Cm,gl,Vl,tau_ref,Iext):
#I am creating all the populations in each layer
pops=[[],[],[],[]]
for h in range(0,4):
    for z in range(0,4):

        Vth= Vt[h][z]*mV
        Vrest=Vr[h][z]*mV

        pop = NeuronGroup(N[h][z], model=eqs, threshold='v > Vth', reset='v = Vrest', refractory=tau_ref[h][z]*ms, method='euler')

        pop.C_m = Cm[h][z]* pF
        pop.g_m= gl[h][z]*nS
        pop.V_L = Vl[h][z] *mV
        pop.V_I= Vl[h][z] *mV

        #I am using the same for everyone for now
        pop.g_AMPA_ext= 1*nS
        #pop.w_ext= 1 #wext[h][z] # I chose the same for everyone now
        pop.g_AMPA_rec = 1*nS #0.95*nS
        pop.g_NMDA = 1*nS #0.05*nS
        pop.g_GABA = 1*nS

        pop.I_ext= Iext[h][z]* pA
        #pop.I_ext1= Iext1[h][z]* pA #If I want an input to another group


        for k in range(0,int(N[h][z])):
            pop[k].v[0]=Vrest

        pops[h].append(pop)
        del (pop)
#return pops
    
# def create_pop_l1(Nl1,Vt_l1,Vr_l1,Cm_l1,gl_l1,Vl_l1,tau_ref_l1,Iext_l1):
#I create the population in layer 1
Vth_l1= Vt_l1*mV
Vrest_l1=Vr_l1*mV
popl1 = NeuronGroup(Nl1, model=eqs, threshold='v > Vth_l1', reset='v = Vrest_l1', refractory=tau_ref_l1*ms, method='euler')

popl1.C_m = Cm_l1* pF
popl1.g_m= gl_l1*nS
popl1.V_L = Vl_l1 *mV
popl1.V_I = Vl_l1 *mV

popl1.g_AMPA_ext= 1*nS
#popl1.wext= 1
popl1.g_AMPA_rec = 1*nS
popl1.g_NMDA = 1*nS
popl1.g_GABA = 1*nS
popl1.I_ext= Iext_l1* pA

for k in range(0,int(Nl1)):
    popl1[k].v[0]=Vrest_l1  
    
#return popl1
    
#IMPORTANT:
#pops: each row is a layer, each column a different subpopulation
#rows: 0=layer2/3, 1=layer4, 2=layer5, 3=layer6
#columns 0=e 1=pv 2=sst 3=vip

# pops=create_populations(N,eqs,Vt,Vr,Cm,gl,Vl,tau_ref,Iext)
# popl1=create_pop_l1(Nl1,Vt_l1,Vr_l1,Cm_l1,gl_l1,Vl_l1,tau_ref_l1,Iext_l1)
end_populations= time.time()


# In[15]:


#I create a poisson generator for each neuron in the population, all the neurons infact are receiving the inputs
#Function to connect each group to the noise
def input_layer_connect(Num,pop,gext,nu_ext): #nu_ext must be in Hz!!
    extinput=PoissonGroup(Num, rates = nu_ext)
    extconn = Synapses(extinput, pop, 'w: 1 ',on_pre='s_AMPA_ext += w')
    extconn.connect(j='i')
    extconn.w= gext 
    return extinput,extconn
#gext=1!!!!! (I already have it in the equations, I keep it so this way I could change it for the different popupaltions in the future)


# In[16]:


start_noise_conn= time.time()
#I call each time the function to connect each group to noise and save the connections (needed for Brian simulation)
print("Connecting noise devices to pupulations")
gext=1
#nu_ext=np.loadtxt('nu_ext.txt') # Is at the beginning
all_extinput=[]
all_extconn=[]

extinput_23e,extconn_23e=input_layer_connect(N[0][0],pops[0][0],gext,nu_ext[0][0]* Hz) #Connecting all populations of layer 2/3 to noise
extinput_23pv,extconn_23pv=input_layer_connect(N[0][1],pops[0][1],gext,nu_ext[0][1]* Hz) #Connecting all populations of layer 2/3 to noise
extinput_23sst,extconn_23sst=input_layer_connect(N[0][2],pops[0][2],gext,nu_ext[0][2]* Hz) #Connecting all populations of layer 2/3 to noise
extinput_23vip,extconn_23vip=input_layer_connect(N[0][3],pops[0][3],gext,nu_ext[0][3]* Hz) #Connecting all populations of layer 2/3 to noise

all_extinput.append(extinput_23e)
all_extinput.append(extinput_23pv)
all_extinput.append(extinput_23sst)
all_extinput.append(extinput_23vip)

all_extconn.append(extconn_23e)
all_extconn.append(extconn_23pv)
all_extconn.append(extconn_23sst)
all_extconn.append(extconn_23vip)

del extinput_23e,extinput_23pv,extinput_23sst,extinput_23vip
del extconn_23e,extconn_23pv,extconn_23sst,extconn_23vip

extinput_4e,extconn_4e=input_layer_connect(N[1][0],pops[1][0],gext,nu_ext[1][0]* Hz) #Connecting all populations of layer 2/3 to noise
extinput_4pv,extconn_4pv=input_layer_connect(N[1][1],pops[1][1],gext,nu_ext[1][1]* Hz) #Connecting all populations of layer 2/3 to noise
extinput_4sst,extconn_4sst=input_layer_connect(N[1][2],pops[1][2],gext,nu_ext[1][2]* Hz) #Connecting all populations of layer 2/3 to noise
extinput_4vip,extconn_4vip=input_layer_connect(N[1][3],pops[1][3],gext,nu_ext[1][3]* Hz) #Connecting all populations of layer 2/3 to noise

all_extinput.append(extinput_4e)
all_extinput.append(extinput_4pv)
all_extinput.append(extinput_4sst)
all_extinput.append(extinput_4vip)

all_extconn.append(extconn_4e)
all_extconn.append(extconn_4pv)
all_extconn.append(extconn_4sst)
all_extconn.append(extconn_4vip)

del extinput_4e,extinput_4pv,extinput_4sst,extinput_4vip
del extconn_4e,extconn_4pv,extconn_4sst,extconn_4vip

extinput_5e,extconn_5e=input_layer_connect(N[2][0],pops[2][0],gext,nu_ext[2][0]* Hz) #Connecting all populations of layer 2/3 to noise
extinput_5pv,extconn_5pv=input_layer_connect(N[2][1],pops[2][1],gext,nu_ext[2][1]* Hz) #Connecting all populations of layer 2/3 to noise
extinput_5sst,extconn_5sst=input_layer_connect(N[2][2],pops[2][2],gext,nu_ext[2][2]* Hz) #Connecting all populations of layer 2/3 to noise
extinput_5vip,extconn_5vip=input_layer_connect(N[2][3],pops[2][3],gext,nu_ext[2][3]* Hz) #Connecting all populations of layer 2/3 to noise

all_extinput.append(extinput_5e)
all_extinput.append(extinput_5pv)
all_extinput.append(extinput_5sst)
all_extinput.append(extinput_5vip)

all_extconn.append(extconn_5e)
all_extconn.append(extconn_5pv)
all_extconn.append(extconn_5sst)
all_extconn.append(extconn_5vip)

del extinput_5e,extinput_5pv,extinput_5sst,extinput_5vip
del extconn_5e,extconn_5pv,extconn_5sst,extconn_5vip


extinput_6e,extconn_6e=input_layer_connect(N[3][0],pops[3][0],gext,nu_ext[3][0]* Hz) #Connecting all populations of layer 2/3 to noise
extinput_6pv,extconn_6pv=input_layer_connect(N[3][1],pops[3][1],gext,nu_ext[3][1]* Hz) #Connecting all populations of layer 2/3 to noise
extinput_6sst,extconn_6sst=input_layer_connect(N[3][2],pops[3][2],gext,nu_ext[3][2]* Hz) #Connecting all populations of layer 2/3 to noise
extinput_6vip,extconn_6vip=input_layer_connect(N[3][3],pops[3][3],gext,nu_ext[3][3]* Hz) #Connecting all populations of layer 2/3 to noise

all_extinput.append(extinput_6e)
all_extinput.append(extinput_6pv)
all_extinput.append(extinput_6sst)
all_extinput.append(extinput_6vip)


all_extconn.append(extconn_6e)
all_extconn.append(extconn_6pv)
all_extconn.append(extconn_6sst)
all_extconn.append(extconn_6vip)

del extinput_6e,extinput_6pv,extinput_6sst,extinput_6vip
del extconn_6e,extconn_6pv,extconn_6sst,extconn_6vip
end_noise_conn= time.time()    
#I HAD TO CREATE A LIST! TO THEN SAVE EVERYTHING
#Brian needs all these included in Network before the simulation starts (see later)


#Connect L1 to noise
extinput_1,extconn_1=input_layer_connect(Nl1,popl1,gext,nu_extl1) #Connecting vipL1 to noise

all_extinput.append(extinput_1)
all_extconn.append(extconn_1)

del extinput_1,extconn_1


# In[17]:


# def connect_populations(lista sorgenti, lista di arrivo, flag nmda, matrice dei pesi, matrice probavbilità,
#         matrice degli N, popolazioni):
#             for sulle sorgenti
#                 for sugli arrivi
#                     attivare ampa o gama in base al tipo di neurone
#                     settare i parametri
                    
#                     attivare nmda


# In[18]:


#sources=[[layer,cell_type],[layer,cell_type]] #sources[k][1] is the cell type
                                                #sources[k][0] is the layer


# In[19]:


def connect_populations(sources,targets,G,Cs,Cp,N,pops,d,e_ampa,e_nmda,i_ampa,i_nmda,nmda_on=True):
    
    #percentage of different receptors, I multiply the prob of connection by this. (Passed in the function)
#     e_ampa=0.8
#     e_nmda=0.2
#     i_ampa=0.8
#     i_nmda=0.2
    
    All_C=[] 
    wp_p=1
    wp_m=1
    for h in range(len(sources)):
        for k in range(len(targets)):
            s_layer = sources[h][0]
            s_cell_type = sources[h][1]
            t_layer = targets[k][0]
            t_cell_type = targets[k][1]
            
            if s_cell_type==0: #0 is excitatory neuron
                
                
                if t_cell_type==0: # excitatory neuron
                
                    conn= Synapses(pops[s_layer][s_cell_type],pops[t_layer][t_cell_type],model=eqs_ampa[s_layer],
                                   on_pre='''
                                   s_AMPA+= 1
                                   apre += dApre
                                   w_AMPA=w_AMPA+apost #w = clip(w + apost, 0, wmax)
                                   ''',
                                   on_post='''
                                   apost += dApost
                                   w_AMPA=w_AMPA + apre #w = clip(w + apre, 0, wmax)
                                   ''', method='euler')
                    conn.connect(condition='i != j',p=e_ampa*Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type])
                    #conn.connect(condition='i != j',p=Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*1)# when NMDA off use this

                    if s_layer==t_layer and s_cell_type==t_cell_type:
                        wp=wp_p
                    else:
                        wp=wp_m
                    #print("Printing the connections")
                    #print(conn.N_outgoing_pre)
                    if Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]==0:
                        conn.w_AMPA= 0
                    else:
                        conn.w_AMPA= wp* G*Cs[4*s_layer+s_cell_type][4*t_layer+t_cell_type]/(e_ampa*Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*N[s_layer][s_cell_type])
                        #conn.w_syn=1
                    conn.delay='d'
                    All_C.append(conn)
                    del conn

                    if nmda_on==True:
                        conn1= Synapses(pops[s_layer][s_cell_type],pops[t_layer][t_cell_type],model=eqs_nmda[s_layer],
                                        on_pre='''
                                        x+=1
                                        apre += dApre
                                        w_NMDA=w_NMDA+apost #w = clip(w + apost, 0, wmax)
                                        ''',
                                        on_post='''
                                        apost += dApost
                                        w_NMDA=w_NMDA + apre #w = clip(w + apre, 0, wmax)
                                        ''', method='euler')
                                        
                        conn1.connect(condition='i != j',p=e_nmda*Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type])
                        #conn1.connect(condition='i != j',p=Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*1) # when I try weights instead
                        if Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]==0:
                            conn1.w_NMDA= 0
                        else:
                            conn1.w_NMDA= wp*  G* Cs[4*s_layer+s_cell_type][4*t_layer+t_cell_type]/(e_nmda*Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*N[s_layer][s_cell_type])
                            #conn1.w_synN=1
                        conn1.delay='d'
                        All_C.append(conn1)
                        del conn1
                        
                if t_cell_type!=0: # inhibitory neuron
                
                    conn= Synapses(pops[s_layer][s_cell_type],pops[t_layer][t_cell_type],model=eqs_ampa[s_layer],on_pre='s_AMPA+=1', method='euler')
                    conn.connect(condition='i != j',p=i_ampa*Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type])
                    #conn.connect(condition='i != j',p=Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*1)# when NMDA off use this

                    if s_layer==t_layer and s_cell_type==t_cell_type:
                        wp=wp_p
                    else:
                        wp=wp_m
                    #print("Printing the connections")
                    #print(conn.N_outgoing_pre)
                    if Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]==0:
                        conn.w_AMPA= 0
                    else:
                        conn.w_AMPA= wp* G*Cs[4*s_layer+s_cell_type][4*t_layer+t_cell_type]/(i_ampa*Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*N[s_layer][s_cell_type])

                    conn.delay='d'
                    All_C.append(conn)
                    del conn

                    if nmda_on==True:
                        conn1= Synapses(pops[s_layer][s_cell_type],pops[t_layer][t_cell_type],model=eqs_nmda[s_layer],on_pre='x+=1', method='euler')
                        conn1.connect(condition='i != j',p=i_nmda*Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type])
                        #conn1.connect(condition='i != j',p=Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*1) # when I try weights instead
                        if Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]==0:
                            conn1.w_NMDA= 0
                        else:
                            conn1.w_NMDA= wp*  G* Cs[4*s_layer+s_cell_type][4*t_layer+t_cell_type]/(i_nmda*Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*N[s_layer][s_cell_type])

                        conn1.delay='d'
                        All_C.append(conn1)
                        del conn1
                
            else:
                conn2= Synapses(pops[s_layer][s_cell_type],pops[t_layer][t_cell_type],model=eqs_gaba[3*s_layer+s_cell_type-1],on_pre='s_GABA+=1', method='euler')
                conn2.connect(condition='i != j',p=Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type])
                if s_layer==t_layer and s_cell_type==t_cell_type:
                    wp=wp_p
                else:
                    wp=wp_m
                
                if Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]==0:
                    conn2.w_GABA= 0
                else:
                    conn2.w_GABA= wp*  G* Cs[4*s_layer+s_cell_type][4*t_layer+t_cell_type]/(Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*N[s_layer][s_cell_type])
#                 print(" Sending: layer=%s, type=%s"%(s_layer,s_cell_type))
#                 print(" Receiving: layer=%s, type=%s"%(t_layer,t_cell_type))
#                 print("Printing the connections sent")
#                 print(conn2.N_outgoing_pre) 
#                 print("Printing the connections arriving")
#                 print(conn2.N_incoming_post) 
                conn2.delay='d'
                All_C.append(conn2)
                del conn2
                
    return All_C
# I have to assign to each source his own equation bijectively. This eqs_gaba[3*s_layer+s_cell_type-1]
#trasform the pair [layer][cell_type] into a number corresponding to one of the 11 gaba equations


# I want a correspondace between my matrix 4x4 (layer,cell type) and the matrix 16x16 where all the values of the connections are stored.
# This is why I need #Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]

#Three types of possibile w: recaled or not, sqrt, /Ntot


# In[20]:


#Function to connect l1 to populations
def connect_l1to_target(targets,Gl1,Csl1,Cpl1,Nl1,pops,popl1,d):
    All_C_l1=[] 
    for k in range(len(targets)):
            t_layer = targets[k][0]
            t_cell_type = targets[k][1]
    
            conn2= Synapses(popl1,pops[t_layer][t_cell_type],model=eqs_gaba_l1,on_pre='s_GABA+=1', method='euler')
            conn2.connect(condition='i != j',p=Cpl1[4*t_layer+t_cell_type])

            if Csl1[4*t_layer+t_cell_type]==0 or Cpl1[4*t_layer+t_cell_type]==0:
                conn2.w_GABA= 0 
            else:
                conn2.w_GABA= Gl1* Csl1[4*t_layer+t_cell_type]/(Cpl1[4*t_layer+t_cell_type]*Nl1)
            conn2.delay='d'
            All_C_l1.append(conn2)
            del conn2
    return All_C_l1


# In[21]:


#Function to connect l1 to l1
def connect_l1_l1(Gl1,Cs_l1_l1,Cp_l1_l1,Nl1,popl1,d):
    conn2= Synapses(popl1,popl1,model=eqs_gaba_l1,on_pre='s_GABA+=1', method='euler')
    conn2.connect(condition='i != j',p=Cp_l1_l1)
    #conn2.w_GABA= Cs_l1_l1
    conn2.w_GABA= Gl1* Cs_l1_l1/(Cp_l1_l1*Nl1)
    conn2.delay='d'
    return conn2


# In[22]:


#Function to connect populations to l1
def connect_source_tol1(sources,Gl1,Cs_tol1,Cp_tol1,N,pops,popl1,d,i_ampa,i_nmda,nmda_on=True):
    All_C=[] 
    for h in range(len(sources)):
        s_layer = sources[h][0]
        s_cell_type = sources[h][1]

        if s_cell_type==0: #0 is excitatory neuron
            conn= Synapses(pops[s_layer][s_cell_type],popl1,model=eqs_ampa[s_layer],on_pre='s_AMPA+=1', method='euler')
            conn.connect(condition='i != j',p=Cp_tol1[4*s_layer+s_cell_type]*i_ampa)

            #print(conn.N_outgoing_pre)
            if Cs_tol1[4*s_layer+s_cell_type]==0 or Cp_tol1[4*s_layer+s_cell_type]:
                conn.w_AMPA= 0
            else:
                conn.w_AMPA= Gl1* Cs_tol1[4*s_layer+s_cell_type]/(i_ampa*Cp_tol1[4*s_layer+s_cell_type]*N[s_layer][s_cell_type])

            conn.delay='d'
            All_C.append(conn)
            del conn

            if nmda_on==True:
                conn1= Synapses(pops[s_layer][s_cell_type],popl1,model=eqs_nmda[s_layer],on_pre='x+=1', method='euler')
                conn1.connect(condition='i != j',p=Cp_tol1[4*s_layer+s_cell_type]*i_nmda)
                if Cs_tol1[4*s_layer+s_cell_type]==0 or Cp_tol1[4*s_layer+s_cell_type]:
                    conn1.w_NMDA= 0
                else:
                    conn1.w_NMDA=Gl1*  Cs_tol1[4*s_layer+s_cell_type]/(i_nmda*Cp_tol1[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*N[s_layer][s_cell_type])

                conn1.delay='d'
                All_C.append(conn1)
                del conn1
        else:
            conn2= Synapses(pops[s_layer][s_cell_type],popl1,model=eqs_gaba[3*s_layer+s_cell_type-1],on_pre='s_GABA+=1', method='euler')
            conn2.connect(condition='i != j',p=Cp_tol1[4*s_layer+s_cell_type])

            if Cs_tol1[4*s_layer+s_cell_type]==0 or Cp_tol1[4*s_layer+s_cell_type]:
                conn2.w_GABA= 0
            else:
                conn2.w_GABA=Gl1* Cs_tol1[4*s_layer+s_cell_type]/(Cp_tol1[4*s_layer+s_cell_type]*N[s_layer][s_cell_type])

            conn2.delay='d'
            All_C.append(conn2)
            del conn2

    return All_C
# I have to assign to each source his own equation bijectively. This eqs_gaba[3*s_layer+s_cell_type-1]
#trasform the pair [layer][cell_type] into a number corresponding to one of the 11 gaba equations


# I want a correspondace between my matrix 4x4 (layer,cell type) and the matrix 16x16 where all the values of the connections are stored.
# This is why I need #Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]

#Three types of possibile w: recaled or not, sqrt, /Ntot


# In[23]:


#CONNECTING ALL LAYERS
def connect_all_layers(Cs,Cp,G,N,pops,d,e_ampa,e_nmda,i_ampa,i_nmda,nmda_on=True):
    targets=[[0,0],[0,1],[0,2],[0,3],
            [1,0],[1,1],[1,2],[1,3],
            [2,0],[2,1],[2,2],[2,3],
            [3,0],[3,1],[3,2],[3,3]]
    sources=[[0,0],[0,1],[0,2],[0,3],
            [1,0],[1,1],[1,2],[1,3],
            [2,0],[2,1],[2,2],[2,3],
            [3,0],[3,1],[3,2],[3,3]]
    conn_all=connect_populations(sources,targets,G,Cs,Cp,N,pops,d,e_ampa,e_nmda,i_ampa,i_nmda,nmda_on=True)
    return conn_all

#CONNECTING 2 LAYERS
def connect_layers(layer_s,layer_t,G,Cs,Cp,N,pops,d,e_ampa,e_nmda,i_ampa,i_nmda,nmda_on=True):
    targets=[[[0,0],[0,1],[0,2],[0,3]],
            [[1,0],[1,1],[1,2],[1,3]],
            [[2,0],[2,1],[2,2],[2,3]],
            [[3,0],[3,1],[3,2],[3,3]]]
    sources=[[[0,0],[0,1],[0,2],[0,3]],
            [[1,0],[1,1],[1,2],[1,3]],
            [[2,0],[2,1],[2,2],[2,3]],
            [[3,0],[3,1],[3,2],[3,3]]]
    
    conn=connect_populations(sources[layer_s],targets[layer_t],G,Cs,Cp,N,pops,d,e_ampa,e_nmda,i_ampa,i_nmda,nmda_on=True)
    return conn

#connection l1 to all
def connect_l1_all(Gl1,Csl1,Cpl1,Cs_tol1,Cp_tol1,Cs_l1_l1,Cp_l1_l1,N,Nl1,pops,popl1,d,i_ampa,i_nmda,nmda_on=True):
    targets=[[0,0],[0,1],[0,2],[0,3],
            [1,0],[1,1],[1,2],[1,3],
            [2,0],[2,1],[2,2],[2,3],
            [3,0],[3,1],[3,2],[3,3]]
    sources=[[0,0],[0,1],[0,2],[0,3],
            [1,0],[1,1],[1,2],[1,3],
            [2,0],[2,1],[2,2],[2,3],
            [3,0],[3,1],[3,2],[3,3]]
    
    conn_l1_to_all=connect_l1to_target(targets,Gl1,Csl1,Cpl1,Nl1,pops,popl1,d)
    conn_all_to_l1=connect_source_tol1(sources,Gl1,Cs_tol1,Cp_tol1,N,pops,popl1,d,i_ampa,i_nmda,nmda_on=True)
    conn_l1_l1=[connect_l1_l1(Gl1,Cs_l1_l1,Cp_l1_l1,Nl1,popl1,d)]
    conn= conn_l1_to_all+ conn_all_to_l1 + conn_l1_l1
    return conn

def connect_all_layers_novip(Cs,Cp,G,N,pops,d,e_ampa,e_nmda,i_ampa,i_nmda,nmda_on=True):
    targets=[[0,0],[0,1],[0,2],[0,3],
            [1,0],[1,1],[1,2],
            [2,0],[2,1],[2,2],
            [3,0],[3,1],[3,2]]
    sources=[[0,0],[0,1],[0,2],[0,3],
            [1,0],[1,1],[1,2],
            [2,0],[2,1],[2,2],
            [3,0],[3,1],[3,2]]
    conn_all=connect_populations(sources,targets,G,Cs,Cp,N,pops,d,e_ampa,e_nmda,i_ampa,i_nmda,nmda_on=True)
    return conn_all


# In[24]:


#SIMPLE TESTS
# sources=[[0,1],[0,2],[0,3]]
# targets=[[0,1],[0,2],[0,3]]
# conprova=connect_populations(sources,targets,G Cs,Cp,N,pops,d,nmda_on=True) #e_ampa to insert!
# #print(conprova)
# connections=conprova


# In[25]:


#conn4_4=connect_layers(1,1,G,Cs,Cp,N,pops,d,nmda_on=True)
# conn23_23=connect_layers(0,0,Cs,Cp,G,N,pops,d,nmda_on=True)
# conn23to4=connect_layers(0,1,Cs,Cp,G,N,pops,d,nmda_on=True)
# connections= conn23_23 + conn23to4


# In[26]:


start_connecting=time.time()
print('--------------------------------------------------')
print('Connecting layers')
conn_all_l1=connect_l1_all(Gl1,Csl1,Cpl1,Cs_tol1,Cp_tol1,Cs_l1_l1,Cp_l1_l1,N,Nl1,pops,popl1,d,i_ampa,i_nmda,nmda_on=True)
conn_all=connect_all_layers(Cs,Cp,G,N,pops,d,e_ampa,e_nmda,i_ampa,i_nmda,nmda_on=True)
connections=conn_all+conn_all_l1
#connections=connect_all_layers_novip(Cs,Cp,G,N,pops,d,e_ampa,e_nmda,i_ampa,i_nmda,nmda_on=True)
print('All layers now connected')
end_connecting=time.time()


# In[27]:


#FUNCTIONS TO MONITOR
#spike detectors
def spike_det(pops,layer,rec=True):
    e_spikes = SpikeMonitor(pops[layer][0],record=rec) #create the spike detector for e
    pv_spikes= SpikeMonitor(pops[layer][1],record=rec) #create the spike detector for subgroup pv
    sst_spikes= SpikeMonitor(pops[layer][2],record=rec) #create the spike detector for subgroup sst
    vip_spikes= SpikeMonitor(pops[layer][3],record=rec)#create the spike detector for subgroup vip
    
    return e_spikes,pv_spikes,sst_spikes,vip_spikes

#subgroup
def spike_det_n(pops,layer,n_activity):
    e_spikes = SpikeMonitor(pops[layer][0][:n_activity]) #create the spike detector for e
    pv_spikes= SpikeMonitor(pops[layer][1][:n_activity]) #create the spike detector for subgroup pv
    sst_spikes= SpikeMonitor(pops[layer][2][:n_activity]) #create the spike detector for subgroup sst
    vip_spikes= SpikeMonitor(pops[layer][3][:n_activity])#create the spike detector for subgroup vip
    
    return e_spikes,pv_spikes,sst_spikes,vip_spikes

#rate detectors
def rate_det(pops,layer):
    e_rate = PopulationRateMonitor(pops[layer][0]) #create the rate det for e
    pv_rate= PopulationRateMonitor(pops[layer][1]) #create the rate detector for subgroup pv
    sst_rate= PopulationRateMonitor(pops[layer][2]) #create the rate detector for subgroup sst
    vip_rate= PopulationRateMonitor(pops[layer][3])#create the rate detector for subgroup vip
    
    return e_rate,pv_rate,sst_rate,vip_rate


# In[28]:


#------------------------------------------------------------------------------
# Create the recording devices: Spike detectors and rate detectors (calling the functions)
#------------------------------------------------------------------------------

start_detectors=time.time()

#mE=StateMonitor(pops[0][0][10], 'v',record=True) #check potential of one neuron
#Spike monitor layer 2/3 with all the neurons recorded
# S_e = SpikeMonitor(pops[0][0], record=True) #LAYER 2_3
# S_pv = SpikeMonitor(pops[0][1], record=True)
# S_sst = SpikeMonitor(pops[0][2], record=True)
# S_vip = SpikeMonitor(pops[0][3], record=True)


# n_activity=10 #I record only 10 neuron for each group
# f=open("n_activity.txt",'w+') #create the file
# f.write('%i ' %n_activity)
# f.close()
# #Create the detectors for each desired layer (using the function I created) only for a subgroup
# S_e23,S_pv23,S_sst23,S_vip23= spike_det_n(pops,0,n_activity) #Spike det of pops in layer 2/3
# S_e4,S_pv4,S_sst4,S_vip4= spike_det_n(pops,1,n_activity) #Spike det of pops in layer 4
# S_e5,S_pv5,S_sst5,S_vip5= spike_det_n(pops,2,n_activity) #Spike det of pops in layer 5
# S_e6,S_pv6,S_sst6,S_vip6= spike_det_n(pops,3,n_activity) #Spike det of pops in layer 6
#S_vip1 = SpikeMonitor(popl1[:n_activity])

#detectors layer1 
S_vip1 = SpikeMonitor(popl1[:],record=True)
R_vip1 = PopulationRateMonitor(popl1)

#Create the detectors for each desired layer (using the function I created) for all neurons
S_e23,S_pv23,S_sst23,S_vip23= spike_det(pops,0,True) #Spike det of pops in layer 2/3
S_e4,S_pv4,S_sst4,S_vip4= spike_det(pops,1,True) #Spike det of pops in layer 4
S_e5,S_pv5,S_sst5,S_vip5= spike_det(pops,2,True) #Spike det of pops in layer 5
S_e6,S_pv6,S_sst6,S_vip6= spike_det(pops,3,True) #Spike det of pops in layer 6

# record instantaneous populations activity
R_e23,R_pv23,R_sst23,R_vip23= rate_det(pops,0) #rate det of pops in layer 2/3
R_e4,R_pv4,R_sst4,R_vip4= rate_det(pops,1) #rate det of pops in layer 4
R_e5,R_pv5,R_sst5,R_vip5= rate_det(pops,2) #rate det of pops in layer 5
R_e6,R_pv6,R_sst6,R_vip6= rate_det(pops,3) #rate det of pops in layer 6

end_detectors=time.time()



# In[29]:


#FOR LFP RECORDINGS I NEED THESE
# num=np.array(np.loadtxt('LFP_files/numberLFP.txt') ) #Array containing the number of neuron in each sub section of the sphere of the electrode
# print(num)

# indeces23=[i for i in range(0,int(num[0]))]
# indeces4=[i for i in range(0,int(num[1]))]
# indeces5=[i for i in range(0,int(num[2]))]
# indeces6=[i for i in range(0,int(num[3]))]

# #I only record the subfractions of neurons which are inside the sphere of resolution of the electrode, info contained in num
# iampaE23=StateMonitor(pops[0][0], 'I_AMPA_rec',record=indeces23)
# igabaE23=StateMonitor(pops[0][0], 'I_GABA',record=indeces23)
# inmdaE23=StateMonitor(pops[0][0], 'I_NMDA',record=indeces23)
# iampaextE23=StateMonitor(pops[0][0], 'I_AMPA_ext',record=indeces23)

# iampaE4=StateMonitor(pops[1][0], 'I_AMPA_rec',record=indeces4)
# igabaE4=StateMonitor(pops[1][0], 'I_GABA',record=indeces4)
# inmdaE4=StateMonitor(pops[1][0], 'I_NMDA',record=indeces4)
# iampaextE4=StateMonitor(pops[1][0], 'I_AMPA_ext',record=indeces4)

# iampaE5=StateMonitor(pops[2][0], 'I_AMPA_rec',record=indeces5)
# igabaE5=StateMonitor(pops[2][0], 'I_GABA',record=indeces5)
# inmdaE5=StateMonitor(pops[2][0], 'I_NMDA',record=indeces5)
# iampaextE5=StateMonitor(pops[2][0], 'I_AMPA_ext',record=indeces5)

# iampaE6=StateMonitor(pops[3][0], 'I_AMPA_rec',record=indeces6)
# igabaE6=StateMonitor(pops[3][0], 'I_GABA',record=indeces6)
# inmdaE6=StateMonitor(pops[3][0], 'I_NMDA',record=indeces6)
# iampaextE6=StateMonitor(pops[3][0], 'I_AMPA_ext',record=indeces6)


# In[30]:


print(connections[80])
#M = StateMonitor(connections[1], ['w_AMPA', 'apre', 'apost'], record=[10])
print(len(connections[:]))
#M = StateMonitor(connections[0], ['w_AMPA', 'apre', 'apost'], record=[10])


# In[31]:


#Calculating the times 
import_time=end_import - startbuild
pop_time=end_populations -start_populations 
noise_time=end_noise_conn -start_noise_conn
connecting_time=end_connecting - start_connecting
detector_time= end_detectors-start_detectors

print('--------------------------------------------------')
print("Import time     : %.2f s = %.2f min " %(import_time,import_time/60))
print("Population time     : %.2f s = %.2f min " %(pop_time,pop_time/60))
print("Noise time     : %.2f s = %.2f min " %(noise_time,noise_time/60))
print("Connecting time     : %.2f s = %.2f min " %(connecting_time,connecting_time/60))
print("Detectors time     : %.2f s = %.2f min " %(detector_time,detector_time/60))
print('------------------------------------------------------------------------')


M = StateMonitor(connections[80], ['w_AMPA', 'apre', 'apost'], record=[10,50,60,100])


#------------------------------------------------------------------------------
# Run the simulation
#------------------------------------------------------------------------------

defaultclock.dt = dt_sim*ms #time step of simulations
#runtime = 800.0 * ms  # total simulation (moved at time at the biginnig of program) 

# construct network   
net = Network(pops[:],popl1,all_extinput[:],all_extconn[:],
              connections[:],
              M,
              #mE,
              S_vip1,R_vip1,
              S_e4,S_pv4,S_sst4,S_vip4,
              S_e5,S_pv5,S_sst5,S_vip5,
              S_e6,S_pv6,S_sst6,S_vip6,
              S_e23,S_pv23,S_sst23,S_vip23,
             R_e23,R_pv23,R_sst23,R_vip23,
             R_e4,R_pv4,R_sst4,R_vip4,
             R_e5,R_pv5,R_sst5,R_vip5,
             R_e6,R_pv6,R_sst6,R_vip6,
#               iampaE23,inmdaE23,igabaE23,iampaextE23,
#              iampaE4,inmdaE4,igabaE4,iampaextE4,
#              iampaE5,inmdaE5,igabaE5,iampaextE5,
#              iampaE6,inmdaE6,igabaE6,iampaextE6
             )  

print('Network is Built')
endbuild = time.time() 

print('Start Simulation')
net.run(runtime)
endsimulate = time.time()        
print('Simulation succeded')
#Compute the build time and simulation time
build_time = endbuild - startbuild
sim_time = endsimulate - endbuild


print('------------------------------------------')
print("Building time    : %.2f s = %.2f min " %(build_time,build_time/60))
print("Simulation time   : %.2f s = %.2f min " % (sim_time,sim_time/60))  


# In[32]:


print(len(M.apre))


# In[33]:



figure(figsize=(4, 8))
subplot(211)
plot(M.t/ms, M.apre[0], label='apre')
plot(M.t/ms, M.apost[0], label='apost')

plot(M.t/ms, M.apre[1], label='apre')
plot(M.t/ms, M.apost[1], label='apost')


plot(M.t/ms, M.apre[2], label='apre')
plot(M.t/ms, M.apost[2], label='apost')


plot(M.t/ms, M.apre[3], label='apre')
plot(M.t/ms, M.apost[3], label='apost')

#plot(M.t/ms, M.apre[1], label='apre1')
#plot(M.t/ms, M.apost[1], label='apost1')

legend()
subplot(212)
plot(M.t/ms, M.w_AMPA[0], label='w_AMPA1')
plot(M.t/ms, M.w_AMPA[1], label='w_AMPA2')
plot(M.t/ms, M.w_AMPA[2], label='w_AMPA3')
plot(M.t/ms, M.w_AMPA[3], label='w_AMPA4')

legend(loc='best')
xlabel('Time (ms)');


# In[34]:


#print(igabaE23.I_GABA[0])


# In[35]:


#I save the currents AMPA,GABA,NMDA of the neurons in files, so then I can compute the LFP

# #layer 2/3
# igabaE23_save=[]
# iampaE23_save=[]
# inmdaE23_save=[]
# iampaextE23_save=[]

# for i in range(0,int(num[0])): #num[0] is the number of neurons I recorded from layer 2/3
#     igabaE23_save.append(igabaE23.I_GABA[i]/pA)
#     iampaE23_save.append(iampaE23.I_AMPA_rec[i]/pA)
#     inmdaE23_save.append(inmdaE23.I_NMDA[i]/pA)
#     iampaextE23_save.append(iampaextE23.I_AMPA_ext[i]/pA)
    
# igabaE23_save=np.array(igabaE23_save)
# iampaE23_save=np.array(iampaE23_save)
# inmdaE23_save=np.array(inmdaE23_save)
# iampaextE23_save=np.array(iampaextE23_save)

# #print(len(igabaE23_save)) #is the number of neurons I recorded from
# #print(len(igabaE23_save[0])) #is the number of steps of the sim, currents over time of neuron 0

# np.save("LFP_files/igabaE23.npy", igabaE23_save)
# np.save("LFP_files/iampaE23.npy", iampaE23_save)
# np.save("LFP_files/inmdaE23.npy", inmdaE23_save)
# np.save("LFP_files/iampaextE23.npy", iampaextE23_save)

# #layer 4
# igabaE4_save=[]
# iampaE4_save=[]
# inmdaE4_save=[]
# iampaextE4_save=[]
# for i in range(0,int(num[1])):
#     igabaE4_save.append(igabaE4.I_GABA[i]/pA)
#     iampaE4_save.append(iampaE4.I_AMPA_rec[i]/pA)
#     inmdaE4_save.append(inmdaE4.I_NMDA[i]/pA)
#     iampaextE4_save.append(iampaextE4.I_AMPA_ext[i]/pA)
    
# igabaE4_save=np.array(igabaE4_save)
# iampaE4_save=np.array(iampaE4_save)
# inmdaE4_save=np.array(inmdaE4_save)
# iampaextE4_save=np.array(iampaextE4_save)

# np.save("LFP_files/igabaE4.npy", igabaE4_save)
# np.save("LFP_files/iampaE4.npy", iampaE4_save)
# np.save("LFP_files/inmdaE4.npy", inmdaE4_save)
# np.save("LFP_files/iampaextE4.npy", iampaextE4_save)

# #layer 5
# igabaE5_save=[]
# iampaE5_save=[]
# inmdaE5_save=[]
# iampaextE5_save=[]
# for i in range(0,int(num[2])):
#     igabaE5_save.append(igabaE5.I_GABA[i]/pA)
#     iampaE5_save.append(iampaE5.I_AMPA_rec[i]/pA)
#     inmdaE5_save.append(inmdaE5.I_NMDA[i]/pA)
#     iampaextE5_save.append(iampaextE5.I_AMPA_ext[i]/pA)
    
# igabaE5_save=np.array(igabaE5_save)
# iampaE5_save=np.array(iampaE5_save)
# inmdaE5_save=np.array(inmdaE5_save)
# iampaextE5_save=np.array(iampaextE5_save)

# np.save("LFP_files/igabaE5.npy", igabaE5_save)
# np.save("LFP_files/iampaE5.npy", iampaE5_save)
# np.save("LFP_files/inmdaE5.npy", inmdaE5_save)
# np.save("LFP_files/iampaextE5.npy", iampaextE5_save)

# #layer 6
# igabaE6_save=[]
# iampaE6_save=[]
# inmdaE6_save=[]
# iampaextE6_save=[]
# for i in range(0,int(num[3])):
#     igabaE6_save.append(igabaE6.I_GABA[i]/pA)
#     iampaE6_save.append(iampaE6.I_AMPA_rec[i]/pA)
#     inmdaE6_save.append(inmdaE6.I_NMDA[i]/pA)
#     iampaextE6_save.append(iampaextE6.I_AMPA_ext[i]/pA)
    
# igabaE6_save=np.array(igabaE6_save)
# iampaE6_save=np.array(iampaE6_save)
# inmdaE6_save=np.array(inmdaE6_save)
# iampaextE6_save=np.array(iampaextE6_save)

# np.save("LFP_files/igabaE6.npy", igabaE6_save)
# np.save("LFP_files/iampaE6.npy", iampaE6_save)
# np.save("LFP_files/inmdaE6.npy", inmdaE6_save)
# np.save("LFP_files/iampaextE6.npy", iampaextE6_save)


# In[36]:


#print(igabaE23.I_GABA[1])


# In[37]:


# print(len(igabaE4.I_GABA))
# print(num[1])


# In[38]:


# igabaE23test=np.load("LFP_files/igabaE23.npy")
# print(igabaE23test[1])


# In[39]:


# fig2 = plt.figure(figsize=(15,7))
# plot(mE.t/ms,mE.v[0],label='e')
# xlabel('time (ms)')
# ylabel('Membran potential V (mV)')
# legend()
# show()


# In[40]:


#I write in files N, runtime, G, dt_sim
#print(np.array(N))
f= open("general_files/N.txt", "w")
for row in np.array(N):
    np.savetxt(f, row)
f.close()

#print(runtime)
f=open("general_files/runtime.txt",'w+') #create the file
f.write('%f ' %runtime)
f.close()
#print(dt_sim)

f=open("general_files/dt_sim.txt",'w+') #create the file
f.write('%f ' %dt_sim)
f.close()

f=open("general_files/G.txt",'w+') #create the file
f.write('%f ' %G)
f.close()


# In[41]:


#I write the total number of spikes for each group in a different file

#layer 1
f=open("Spikes_files/S_vip1numspike.txt",'w+') #create the file
f.write('%f ' %S_vip1.num_spikes)
f.close()

#layer 2/3
f=open("Spikes_files/S_e23numspike.txt",'w+') #create the file
f.write('%f ' %S_e23.num_spikes)
f.close()

f=open("Spikes_files/S_pv23numspike.txt",'w+') #create the file
f.write('%f ' %S_pv23.num_spikes) 
f.close()

f=open("Spikes_files/S_sst23numspike.txt",'w+') #create the file
f.write('%f ' %S_sst23.num_spikes) 
f.close()

f=open("Spikes_files/S_vip23numspike.txt",'w+') #create the file
f.write('%f ' %S_vip23.num_spikes) 
f.close()

#layer4
f=open("Spikes_files/S_e4numspike.txt",'w+') #create the file
f.write('%f ' %S_e4.num_spikes) 
f.close()

f=open("Spikes_files/S_pv4numspike.txt",'w+') #create the file
f.write('%f ' %S_pv4.num_spikes) 
f.close()

f=open("Spikes_files/S_sst4numspike.txt",'w+') #create the file
f.write('%f ' %S_sst4.num_spikes) 
f.close()

f=open("Spikes_files/S_vip4numspike.txt",'w+') #create the file
f.write('%f ' %S_vip4.num_spikes) 
f.close()

#layer 5
f=open("Spikes_files/S_e5numspike.txt",'w+') #create the file
f.write('%f ' %S_e5.num_spikes) 
f.close()

f=open("Spikes_files/S_pv5numspike.txt",'w+') #create the file
f.write('%f ' %S_pv5.num_spikes) 
f.close()

f=open("Spikes_files/S_sst5numspike.txt",'w+') #create the file
f.write('%f ' %S_sst5.num_spikes) 
f.close()

f=open("Spikes_files/S_vip5numspike.txt",'w+') #create the file
f.write('%f ' %S_vip5.num_spikes) 
f.close()

#layer6
f=open("Spikes_files/S_e6numspike.txt",'w+') #create the file
f.write('%f ' %S_e6.num_spikes) 
f.close()

f=open("Spikes_files/S_pv6numspike.txt",'w+') #create the file
f.write('%f ' %S_pv6.num_spikes) 
f.close()

f=open("Spikes_files/S_sst6numspike.txt",'w+') #create the file
f.write('%f ' %S_sst6.num_spikes) 
f.close()

f=open("Spikes_files/S_vip6numspike.txt",'w+') #create the file
f.write('%f ' %S_vip6.num_spikes) 
f.close()


# In[42]:


#I write the spikes in files of every neuron. In one file there are the indeces 
#.i of th neurons emitting spike at the corrisponding time of the other file .t

#layer 1
f=open("Spikes_files/S_vip1i.txt",'w+') #create the file
for i in range(0,len(S_vip1.i)):
    f.write('%i ' %S_vip1.i[i])
    f.write('\n')
f.close()

f=open("Spikes_files/S_vip1t.txt",'w+') #create the file
for i in range(0,len(S_vip1.t)):
    f.write('%f ' %S_vip1.t[i]) 
    f.write('\n')
f.close()

#layer 2/3
f=open("Spikes_files/S_e23i.txt",'w+') #create the file
for i in range(0,len(S_e23.i)):
    f.write('%i ' %S_e23.i[i])
    f.write('\n')
f.close()

f=open("Spikes_files/S_e23t.txt",'w+') #create the file
for i in range(0,len(S_e23.t)):
    f.write('%f ' %S_e23.t[i]) 
    f.write('\n')
f.close()

f=open("Spikes_files/S_pv23i.txt",'w+') #create the file
for i in range(0,len(S_pv23.i)):
    f.write('%i ' %S_pv23.i[i]) 
    f.write('\n')
f.close()

f=open("Spikes_files/S_pv23t.txt",'w+') #create the file
for i in range(0,len(S_pv23.t)):
    f.write('%f ' %S_pv23.t[i]) 
    f.write('\n')
f.close()


f=open("Spikes_files/S_sst23i.txt",'w+') #create the file
for i in range(0,len(S_sst23.i)):
    f.write('%i ' %S_sst23.i[i]) 
    f.write('\n')
f.close()

f=open("Spikes_files/S_sst23t.txt",'w+') #create the file
for i in range(0,len(S_sst23.t)):
    f.write('%f ' %S_sst23.t[i])  
    f.write('\n')
f.close()

f=open("Spikes_files/S_vip23i.txt",'w+') #create the file
for i in range(0,len(S_vip23.i)):
    f.write('%i ' %S_vip23.i[i]) 
    f.write('\n')
f.close()

f=open("Spikes_files/S_vip23t.txt",'w+') #create the file
for i in range(0,len(S_vip23.t)):
    f.write('%f ' %S_vip23.t[i]) 
    f.write('\n')
f.close()

#layer4
f=open("Spikes_files/S_e4i.txt",'w+') #create the file
for i in range(0,len(S_e4.i)):
    f.write('%i ' %S_e4.i[i]) 
    f.write('\n')
f.close()

f=open("Spikes_files/S_e4t.txt",'w+') #create the file
for i in range(0,len(S_e4.t)):
    f.write('%f ' %S_e4.t[i]) 
    f.write('\n')
f.close()

f=open("Spikes_files/S_pv4i.txt",'w+') #create the file
for i in range(0,len(S_pv4.i)):
    f.write('%i ' %S_pv4.i[i]) 
    f.write('\n')
f.close()

f=open("Spikes_files/S_pv4t.txt",'w+') #create the file
for i in range(0,len(S_pv4.t)):
    f.write('%f ' %S_pv4.t[i]) 
    f.write('\n')
f.close()

f=open("Spikes_files/S_sst4i.txt",'w+') #create the file
for i in range(0,len(S_sst4.i)):
    f.write('%i ' %S_sst4.i[i]) 
    f.write('\n')
f.close()

f=open("Spikes_files/S_sst4t.txt",'w+') #create the file
for i in range(0,len(S_sst4.t)):
    f.write('%f ' %S_sst4.t[i]) 
    f.write('\n')
f.close()


f=open("Spikes_files/S_vip4i.txt",'w+') #create the file
for i in range(0,len(S_vip4.i)):
    f.write('%i ' %S_vip4.i[i]) 
    f.write('\n')
f.close()

f=open("Spikes_files/S_vip4t.txt",'w+') #create the file
for i in range(0,len(S_vip4.t)):
    f.write('%f ' %S_vip4.t[i]) 
    f.write('\n')
f.close()

#layer 5
f=open("Spikes_files/S_e5i.txt",'w+') #create the file
for i in range(0,len(S_e5.i)):
    f.write('%i ' %S_e5.i[i]) 
    f.write('\n')
f.close()

f=open("Spikes_files/S_e5t.txt",'w+') #create the file
for i in range(0,len(S_e5.t)):
    f.write('%f ' %S_e5.t[i]) 
    f.write('\n')
f.close()

f=open("Spikes_files/S_pv5i.txt",'w+') #create the file
for i in range(0,len(S_pv5.i)):
    f.write('%i ' %S_pv5.i[i]) 
    f.write('\n')
f.close()

f=open("Spikes_files/S_pv5t.txt",'w+') #create the file
for i in range(0,len(S_pv5.t)):
    f.write('%f ' %S_pv5.t[i]) 
    f.write('\n')
f.close()

f=open("Spikes_files/S_sst5i.txt",'w+') #create the file
for i in range(0,len(S_sst5.i)):
    f.write('%i ' %S_sst5.i[i]) 
    f.write('\n')
f.close()

f=open("Spikes_files/S_sst5t.txt",'w+') #create the file
for i in range(0,len(S_sst5.t)):
    f.write('%f ' %S_sst5.t[i]) 
    f.write('\n')
f.close()


f=open("Spikes_files/S_vip5i.txt",'w+') #create the file
for i in range(0,len(S_vip5.i)):
    f.write('%i ' %S_vip5.i[i]) 
    f.write('\n')
f.close()

f=open("Spikes_files/S_vip5t.txt",'w+') #create the file
for i in range(0,len(S_vip5.t)):
    f.write('%f ' %S_vip5.t[i]) 
    f.write('\n')
f.close()


#layer6
f=open("Spikes_files/S_e6i.txt",'w+') #create the file
for i in range(0,len(S_e6.i)):
    f.write('%i ' %S_e6.i[i]) 
    f.write('\n')
f.close()

f=open("Spikes_files/S_e6t.txt",'w+') #create the file
for i in range(0,len(S_e6.t)):
    f.write('%f ' %S_e6.t[i]) 
    f.write('\n')
f.close()

f=open("Spikes_files/S_pv6i.txt",'w+') #create the file
for i in range(0,len(S_pv6.i)):
    f.write('%i ' %S_pv6.i[i]) 
    f.write('\n')
f.close()

f=open("Spikes_files/S_pv6t.txt",'w+') #create the file
for i in range(0,len(S_pv6.t)):
    f.write('%f ' %S_pv6.t[i]) 
    f.write('\n')
f.close()

f=open("Spikes_files/S_sst6i.txt",'w+') #create the file
for i in range(0,len(S_sst6.i)):
    f.write('%i ' %S_sst6.i[i]) 
    f.write('\n')
f.close()

f=open("Spikes_files/S_sst6t.txt",'w+') #create the file
for i in range(0,len(S_sst6.t)):
    f.write('%f ' %S_sst6.t[i]) 
    f.write('\n')
f.close()


f=open("Spikes_files/S_vip6i.txt",'w+') #create the file
for i in range(0,len(S_vip6.i)):
    f.write('%i ' %S_vip6.i[i]) 
    f.write('\n')
f.close()

f=open("Spikes_files/S_vip6t.txt",'w+') #create the file
for i in range(0,len(S_vip6.t)):
    f.write('%f ' %S_vip6.t[i]) 
    f.write('\n')
f.close()


# In[43]:


# #TO WRTITE THE FILE WITH ACTIVITY!
# f = open("activity_visALLinput.csv", "w")
# for i in range(0,len(i_e23)):
#     f.write('%i,%f\n'%(i_e23[i],np.array(S_e23.t/ms)[i]))
# for i in range(0,len(i_pv23)):
#     f.write('%i,%f\n'%(i_pv23[i],np.array(S_pv23.t/ms)[i]))
# for i in range(0,len(i_sst23)):
#     f.write('%i,%f\n'%(i_sst23[i],np.array(S_sst23.t/ms)[i]))
# for i in range(0,len(i_vip23)):
#     f.write('%i,%f\n'%(i_vip23[i],np.array(S_vip23.t/ms)[i]))
    

# for i in range(0,len(i_e4)):
#     f.write('%i,%f\n'%(i_e4[i]+N2_3,np.array(S_e4.t/ms)[i]))
# for i in range(0,len(i_pv4)):
#     f.write('%i,%f\n'%(i_pv4[i]+N2_3,np.array(S_pv4.t/ms)[i]))
# for i in range(0,len(i_sst4)):
#     f.write('%i,%f\n'%(i_sst4[i]+N2_3,np.array(S_sst4.t/ms)[i]))
# for i in range(0,len(i_vip4)):
#     f.write('%i,%f\n'%(i_vip4[i]+N2_3,np.array(S_vip4.t/ms)[i]))    
    


# for i in range(0,len(i_e5)):
#     f.write('%i,%f\n'%(i_e5[i]+N2_3+N4,np.array(S_e5.t/ms)[i]))
# for i in range(0,len(i_pv5)):
#     f.write('%i,%f\n'%(i_pv5[i]+N2_3+N4,np.array(S_pv5.t/ms)[i]))
# for i in range(0,len(i_sst5)):
#     f.write('%i,%f\n'%(i_sst5[i]+N2_3+N4,np.array(S_sst5.t/ms)[i]))
# for i in range(0,len(i_vip5)):
#     f.write('%i,%f\n'%(i_vip5[i]+N2_3+N4,np.array(S_vip5.t/ms)[i]))    
        
# for i in range(0,len(i_e6)):
#     f.write('%i,%f\n'%(i_e6[i]+N2_3+N4+N5,np.array(S_e6.t/ms)[i]))
# for i in range(0,len(i_pv6)):
#     f.write('%i,%f\n'%(i_pv6[i]+N2_3+N4+N5,np.array(S_pv6.t/ms)[i]))
# for i in range(0,len(i_sst6)):
#     f.write('%i,%f\n'%(i_sst6[i]+N2_3+N4+N5,np.array(S_sst6.t/ms)[i]))
# for i in range(0,len(i_vip6)):
#     f.write('%i,%f\n'%(i_vip6[i]+N2_3+N4+N5,np.array(S_vip6.t/ms)[i]))    
        
        
# f.close()


# In[44]:


#CHECKING IF I WROTE EVERYTHING CORRECTLY
# import numpy as np

# my_csv = np.genfromtxt('activity_vis.csv', delimiter=',')
# index = my_csv[:, 0]
# spike_time = my_csv[:, 1]
# print(index)
# print(spike_time)


# In[45]:


# print(len(R_vip1.rate))
# print(len(R_e23.t))
# print(R_vip1.rate/Hz)


# In[46]:


#Write to files the rates.
#In each file I have the rate of the population at each time step.

#layer1
f=open("Rate_files/R_vip1rate.txt",'w+') #create the file
for i in range(0,len(R_vip1.rate)):
    f.write('%f ' %R_vip1.rate[i])
    f.write('\n')
f.close()

#layer 2/3
f=open("Rate_files/R_e23rate.txt",'w+') #create the file
for i in range(0,len(R_e23.rate)):
    f.write('%f ' %R_e23.rate[i])
    f.write('\n')
f.close()

f=open("Rate_files/R_pv23rate.txt",'w+') #create the file
for i in range(0,len(R_pv23.rate)):
    f.write('%f ' %R_pv23.rate[i]) 
    f.write('\n')
f.close()

f=open("Rate_files/R_sst23rate.txt",'w+') #create the file
for i in range(0,len(R_sst23.rate)):
    f.write('%f ' %R_sst23.rate[i]) 
    f.write('\n')
f.close()

f=open("Rate_files/R_vip23rate.txt",'w+') #create the file
for i in range(0,len(R_vip23.rate)):
    f.write('%f ' %R_vip23.rate[i]) 
    f.write('\n')
f.close()

#layer4
f=open("Rate_files/R_e4rate.txt",'w+') #create the file
for i in range(0,len(R_e4.rate)):
    f.write('%f ' %R_e4.rate[i]) 
    f.write('\n')
f.close()

f=open("Rate_files/R_pv4rate.txt",'w+') #create the file
for i in range(0,len(R_pv4.rate)):
    f.write('%f ' %R_pv4.rate[i]) 
    f.write('\n')
f.close()

f=open("Rate_files/R_sst4rate.txt",'w+') #create the file
for i in range(0,len(R_sst4.rate)):
    f.write('%f ' %R_sst4.rate[i]) 
    f.write('\n')
f.close()

f=open("Rate_files/R_vip4rate.txt",'w+') #create the file
for i in range(0,len(R_vip4.rate)):
    f.write('%f ' %R_vip4.rate[i]) 
    f.write('\n')
f.close()

#layer 5
f=open("Rate_files/R_e5rate.txt",'w+') #create the file
for i in range(0,len(R_e5.rate)):
    f.write('%f ' %R_e5.rate[i]) 
    f.write('\n')
f.close()

f=open("Rate_files/R_pv5rate.txt",'w+') #create the file
for i in range(0,len(R_pv5.rate)):
    f.write('%f ' %R_pv5.rate[i]) 
    f.write('\n')
f.close()

f=open("Rate_files/R_sst5rate.txt",'w+') #create the file
for i in range(0,len(R_sst5.rate)):
    f.write('%f ' %R_sst5.rate[i]) 
    f.write('\n')
f.close()

f=open("Rate_files/R_vip5rate.txt",'w+') #create the file
for i in range(0,len(R_vip5.rate)):
    f.write('%f ' %R_vip5.rate[i]) 
    f.write('\n')
f.close()

#layer6
f=open("Rate_files/R_e6rate.txt",'w+') #create the file
for i in range(0,len(R_e6.rate)):
    f.write('%f ' %R_e6.rate[i]) 
    f.write('\n')
f.close()

f=open("Rate_files/R_pv6rate.txt",'w+') #create the file
for i in range(0,len(R_pv6.rate)):
    f.write('%f ' %R_pv6.rate[i]) 
    f.write('\n')
f.close()

f=open("Rate_files/R_sst6rate.txt",'w+') #create the file
for i in range(0,len(R_sst6.rate)):
    f.write('%f ' %R_sst6.rate[i]) 
    f.write('\n')
f.close()

f=open("Rate_files/R_vip6rate.txt",'w+') #create the file
for i in range(0,len(R_vip6.rate)):
    f.write('%f ' %R_vip6.rate[i]) 
    f.write('\n')
f.close()


# In[47]:


#function of the rate plots with a given window, brian ( in the file I don't have this feature included, I only have the rate at each time step, not smussed)
def rate_plots_noI(R_vip1,R_e23,R_pv23,R_sst23,R_vip23,
             R_e4,R_pv4,R_sst4,R_vip4,
             R_e5,R_pv5,R_sst5,R_vip5,
             R_e6,R_pv6,R_sst6,R_vip6,wid):
    


    
    #wid=50LFP_all23=LFP_allNeu(igabaE23,iampaE23,inmdaE23,iampaextE23,igabaPv23,iampaPv23,inmdaPv23,iampaextPv23,igabaSst23,iampaSst23,inmdaSst23,iampaextSst23,igabaVip23,iampaVip23,inmdaVip23,iampaextVip23,0,steps,tau,alpha)
    fig = plt.figure(figsize=(6,4))
    plt.plot(R_vip1.t/ms,R_vip1.smooth_rate(width=25 * ms) / Hz,color='orange', label='vip')
    plt.xlabel('time (ms)')
    plt.xlabel('spikes/s')
    plt.legend()
    plt.title('Pops activity - layer 1')
    plt.show()


    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(15,9))
    f.suptitle('N=%sk, G=%s, B=%s'%(int(Ntot/1000),G,nu_ext),fontsize=10)
    ax1.plot(R_e23.t / ms, R_e23.smooth_rate(width=wid* ms) / Hz,color='r', label='e')
    ax1.plot(R_pv23.t / ms, R_pv23.smooth_rate(width=wid* ms) / Hz,color='b', label='pv')
    ax1.plot(R_sst23.t / ms, R_sst23.smooth_rate(width=wid * ms) / Hz,color='g', label='sst')
    ax1.plot(R_vip23.t / ms, R_vip23.smooth_rate(width=wid * ms) / Hz,color='orange', label='vip')
    ax1.legend()
    ax1.set_xlabel('time (ms)')
    #ax1.set_xlim(300,800)
    #ax1.set_ylim(0,10)
    ax1.set_ylabel('spikes/s')
    ax1.set_title('Population rates - layer 2/3')
    

    ax2.plot(R_e4.t / ms, R_e4.smooth_rate(width=wid * ms) / Hz,color='r', label='e')
    ax2.plot(R_pv4.t / ms, R_pv4.smooth_rate(width=wid * ms) / Hz,color='b', label='pv')
    ax2.plot(R_sst4.t / ms, R_sst4.smooth_rate(width=wid * ms) / Hz,color='g', label='sst')
    ax2.plot(R_vip4.t / ms, R_vip4.smooth_rate(width=wid * ms) / Hz,color='orange', label='vip')
    ax2.set_title('Population rates - layer 4')
    #ax2.set_xlim(300,800)
    #ax2.set_ylim(0,10)
    ax2.set_xlabel('time (ms)')
    ax2.set_ylabel('spikes/s')
    ax2.legend()
    

    ax3.plot(R_e5.t / ms, R_e5.smooth_rate(width=wid * ms) / Hz,color='r', label='e')
    ax3.plot(R_pv5.t / ms, R_pv5.smooth_rate(width=wid * ms) / Hz,color='b', label='pv')
    ax3.plot(R_sst5.t / ms, R_sst5.smooth_rate(width=wid * ms) / Hz,color='g', label='sst')
    ax3.plot(R_vip5.t / ms, R_vip5.smooth_rate(width=wid * ms) / Hz,color='orange', label='vip')
    ax3.set_title('Population rates - layer 5')
    ax3.set_xlabel('time (ms)')
    #ax3.set_xlim(300,800)
    #ax3.set_ylim(0,20)
    ax3.set_ylabel('spikes/s')
    ax3.legend()
    


    ax4.plot(R_e6.t / ms, R_e6.smooth_rate(width=wid * ms) / Hz,color='r', label='e')
    ax4.plot(R_pv6.t / ms, R_pv6.smooth_rate(width=wid * ms) / Hz,color='b', label='pv')
    ax4.plot(R_sst6.t / ms, R_sst6.smooth_rate(width=wid * ms) / Hz,color='g', label='sst')
    ax4.plot(R_vip6.t / ms, R_vip6.smooth_rate(width=wid * ms) / Hz,color='orange', label='vip')
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

    show()


# In[48]:


wid=50
rate_plots_noI(R_vip1,R_e23,R_pv23,R_sst23,R_vip23,
             R_e4,R_pv4,R_sst4,R_vip4,
             R_e5,R_pv5,R_sst5,R_vip5,
             R_e6,R_pv6,R_sst6,R_vip6,wid)


# In[49]:


#------------------------------------------------------------------------------
# Show results population rates for all groups
#------------------------------------------------------------------------------

def rate_plots(R_vip1,R_e23,R_pv23,R_sst23,R_vip23,
             R_e4,R_pv4,R_sst4,R_vip4,
             R_e5,R_pv5,R_sst5,R_vip5,
             R_e6,R_pv6,R_sst6,R_vip6,wid):
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

  
    
    #wid=50LFP_all23=LFP_allNeu(igabaE23,iampaE23,inmdaE23,iampaextE23,igabaPv23,iampaPv23,inmdaPv23,iampaextPv23,igabaSst23,iampaSst23,inmdaSst23,iampaextSst23,igabaVip23,iampaVip23,inmdaVip23,iampaextVip23,0,steps,tau,alpha)
    fig = plt.figure(figsize=(6,4))
    plt.plot(R_vip1.t/ms,R_vip1.smooth_rate(width=25 * ms) / Hz,color='orange', label='vip')
    plt.xlabel('time (ms)')
    plt.xlabel('spikes/s')
    plt.legend()
    plt.title('Pops activity - layer 1')
    plt.show()


    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(15,9))
    f.suptitle('N=%sk, G=%s, B=%s'%(int(Ntot/1000),G,nu_ext),fontsize=10)
    ax1.plot(R_e23.t / ms, R_e23.smooth_rate(width=wid* ms) / Hz,color='r', label='e')
    ax1.plot(R_pv23.t / ms, R_pv23.smooth_rate(width=wid* ms) / Hz,color='b', label='pv')
    ax1.plot(R_sst23.t / ms, R_sst23.smooth_rate(width=wid * ms) / Hz,color='g', label='sst')
    ax1.plot(R_vip23.t / ms, R_vip23.smooth_rate(width=wid * ms) / Hz,color='orange', label='vip')
    ax1.legend()
    ax1.set_xlabel('time (ms)')
    #ax1.set_xlim(300,800)
    #ax1.set_ylim(0,10)
    ax1.set_ylabel('spikes/s')
    ax1.set_title('Population rates - layer 2/3')
    
    ax1 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax1.set_ylabel('Input to e4 (pA)', color='c')  # we already handled the x-label with ax1
    ax1.plot(x, I, color='c')
    ax1.tick_params(axis='y', labelcolor='c')

    

    ax2.plot(R_e4.t / ms, R_e4.smooth_rate(width=wid * ms) / Hz,color='r', label='e')
    ax2.plot(R_pv4.t / ms, R_pv4.smooth_rate(width=wid * ms) / Hz,color='b', label='pv')
    ax2.plot(R_sst4.t / ms, R_sst4.smooth_rate(width=wid * ms) / Hz,color='g', label='sst')
    ax2.plot(R_vip4.t / ms, R_vip4.smooth_rate(width=wid * ms) / Hz,color='orange', label='vip')
    ax2.set_title('Population rates - layer 4')
    #ax2.set_xlim(300,800)
    #ax2.set_ylim(0,10)
    ax2.set_xlabel('time (ms)')
    ax2.set_ylabel('spikes/s')
    ax2.legend()
    
    ax2 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Input to e4 (pA)', color='c')  # we already handled the x-label with ax1
    ax2.plot(x, I, color='c')
    ax2.tick_params(axis='y', labelcolor='c')


    ax3.plot(R_e5.t / ms, R_e5.smooth_rate(width=wid * ms) / Hz,color='r', label='e')
    ax3.plot(R_pv5.t / ms, R_pv5.smooth_rate(width=wid * ms) / Hz,color='b', label='pv')
    ax3.plot(R_sst5.t / ms, R_sst5.smooth_rate(width=wid * ms) / Hz,color='g', label='sst')
    ax3.plot(R_vip5.t / ms, R_vip5.smooth_rate(width=wid * ms) / Hz,color='orange', label='vip')
    ax3.set_title('Population rates - layer 5')
    ax3.set_xlabel('time (ms)')
    #ax3.set_xlim(300,800)
    #ax3.set_ylim(0,20)
    ax3.set_ylabel('spikes/s')
    ax3.legend()
    
    ax3 = ax3.twinx()  # instantiate a second axes that shares the same x-axis
    ax3.set_ylabel('Input to e4 (pA)', color='c')  # we already handled the x-label with ax1
    ax3.plot(x, I, color='c')
    ax3.tick_params(axis='y', labelcolor='c')


    ax4.plot(R_e6.t / ms, R_e6.smooth_rate(width=wid * ms) / Hz,color='r', label='e')
    ax4.plot(R_pv6.t / ms, R_pv6.smooth_rate(width=wid * ms) / Hz,color='b', label='pv')
    ax4.plot(R_sst6.t / ms, R_sst6.smooth_rate(width=wid * ms) / Hz,color='g', label='sst')
    ax4.plot(R_vip6.t / ms, R_vip6.smooth_rate(width=wid * ms) / Hz,color='orange', label='vip')
    ax4.set_title('Population rates - layer 6' )
    ax4.set_xlabel('time (ms)')
    #ax4.set_xlim(300,800)
    #ax4.set_ylim(0,20)
    ax4.set_ylabel('spikes/s')
    ax4.legend()

    ax4 = ax4.twinx()  # instantiate a second axes that shares the same x-axis
    ax4.set_ylabel('Input to e4 (pA)', color='c')  # we already handled the x-label with ax1
    ax4.plot(x, I, color='c')
    ax4.tick_params(axis='y', labelcolor='c')

    
    plt.subplots_adjust(left=0.125,
                        bottom=0.125, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.2, 
                        hspace=0.3)              

    show()


# In[50]:


# wid=50
# rate_plots(R_vip1,R_e23,R_pv23,R_sst23,R_vip23,
#              R_e4,R_pv4,R_sst4,R_vip4,
#              R_e5,R_pv5,R_sst5,R_vip5,
#              R_e6,R_pv6,R_sst6,R_vip6,wid)


# In[ ]:




