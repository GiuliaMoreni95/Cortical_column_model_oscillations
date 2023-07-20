#!/usr/bin/env python
# coding: utf-8

# In[1]:

#final version
#This program simulate my netowork and save in files all the date produced by it.

# In[2]:

from brian2 import *
from brian2tools import *
import numpy as np
import time
import random
import matplotlib.pyplot as plt
# In[3]:
#np.random.seed(20)
#clear_cache('cython')

# In[4]:

startbuild = time.time()
print("Initializing and building the Network")


# In[5]:

######Parameters to change in the simulation###############
a='Sp_input_1_2' #This is the folder taht will contain the spikes
#b='Rt_input2' #This is the folder taht will contain the rates
nu_ext_file='import_files/nu_ext.txt' #This is the file that contains the background noise to the neurons


runtime = 3000.0 * ms # How long you want the simulation
dt_sim=0.1 #ms         # Resolution of the simulation
G=5 #global constant to increase all the connections weights
Gl1=5 #global constant to increase all the connections weights in Layer1
Ntot=5000 #Ntot=226562 Allen inst #Total number of neurons in the simulation

#Percentage of AMPA and NMDA receptors in excitatory and inhibitory neurons
e_ampa=0.8
e_nmda=0.2
i_ampa=0.8
i_nmda=0.2


#External input to the neurons
Iext=np.loadtxt('import_files/Iext0.txt') #File that contain the external input
Iext_l1= 0
#Iext=np.loadtxt('import_files/Iext.txt')
Iext1=np.loadtxt('import_files/Iext0.txt')

#Background noise to the neurons
nu_ext=np.loadtxt(nu_ext_file) #I upload the file
nu_extl1= 650 *Hz

#IF you WANT DIFFERNT receptor weight for each particular group receiving the external background noise
#wext=np.loadtxt('import_files/'+wext_old.txt')
#I CHOSE to have the same weight for all the receptors of the different neurons type receiving the external noise


# In[6]:



N1=int(0.0192574218*Ntot) # N1 is not included in the calculation of Ntot, Ntot is just the 4 layers.
N2_3=int(0.291088453*Ntot) # This percentage are computed looking at the numbers of the Allen institute
N4=int(0.237625904*Ntot)
N5=int(0.17425693*Ntot)
N6= Ntot-N2_3-N4-N5
#N6=int(0.297031276*Ntot)
#print(N2_3+N4+N5+N6)

print("The corticular column in this model is composed by 4 layers+1")
print("Total number of neurons in the column: %s + %s \n85 perc excitatory and 15perc inhibitory \nIn each layer: 1 excitotory population and 3 inhibitory populations: pv, sst and vip cells.   "%(Ntot,N1))


perc_tot=np.loadtxt('import_files/perc_tot.txt') #Matrix containing the percentage of excitatory and inhib neurons
#print(perc_tot)
perc=np.loadtxt('import_files/perc.txt') #Matrix containing the percentage of neurons for each type in each layer
#print(perc)
n_tot= np.array([[N2_3,N2_3,N2_3,N2_3],[N4,N4,N4,N4],[N5,N5,N5,N5],[N6,N6,N6,N6]]) # number of total neuron in each layer,
#I need this n_tot to then be able to create the matrix N which contains the exact number of each neurons for this simulation
#print(n_tot)

N=perc*perc_tot*n_tot #Matrix containing numbers of neurons for each type in each layer
N=np.matrix.round(N)  #I round it to the nearest value integer
N=N.astype(int) # Number of neurons should be of type int
#Now I correct the matrix I obtained,
#the sum of each layer should return the total number of neurons in that layer
for k in range(0,4):
    N[k][0]+= n_tot[k][0]-sum(N[k]) #sum(N[k]) is the total number of neurons in that layer  I have in the matrix
                                    #n_tot[k][0] is the number of neuorons in that layer from the percentages
                                    #If the two numbers don't match the N[k][0] (excitaotry in each layer) gets updated
print("Ntot of 4layers:")
print(sum(N))
print("+ in layer 1 we have:%s neurons"%N1)
print(N)
print('--------------------------------------------------')

# In[7]:

# #FOR THE MOMENT I AM NOT USING DISTINCTION BETWEEN E and I
#they are all 1 nS (see in the equations definition later)
#In the future this can be changed, we could use values from Wang or from the code online.
#I put here all the values the two codes have

# # Connectivity - external connections
# g_AMPA_ext_E = 2.08 * nS #2.1 * nS
# g_AMPA_ext_I = 1.62 * nS

# #ampa connections
# g_AMPA_rec_I = 0.081 * nS # gEI_AMPA = 0.04 * nS    # Weight of excitatory to inhibitory synapses (AMPA)
# g_AMPA_rec_E =0.104 * nS # gEE_AMPA = 0.05 * nS    # Weight of AMPA synapses between excitatory neurons

# # NMDA (excitatory)
# g_NMDA_E = 0.327 * nS # 0.165 * nS # Weight of NMDA synapses between excitatory
# g_NMDA_I =0.258 * nS # 0.13 * nS # Weight of NMDA synapses between excitatory and inhibitory

# # GABAergic (inhibitory)
# g_GABA_E =1.25 * nS # gIE_GABA = 1.3 * nS # Weight of inhibitory to excitatory synapses
# g_GABA_I =0.973 * nS # gII_GABA = 1.0 * nS # Weight of inhibitory to inhibitory synapses

# In[8]:

# Synapse model
w_ext=1                                  #weight for each group for the external background noise going to AMPA, SAME for every population
                                        #This is the Weight in front of s_tot. (see below in eq ampa ext)
                                        #if you don't want it to be the same for every population:
                                        # wext is also in the equations of AMPA ext (see below in eq ampa ext), if needed just uncomment it.
gext=1                                   #how much you affect s_ampa with 1 spike from the Poisson.


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
Cp = np.loadtxt('import_files/connectionsPro.txt') #connenctions probabilities between the 16 groups in the 4 layers (not VIP1)
Cs=np.loadtxt('import_files/connectionsStren.txt') #connenctions strenghts between the 16 groups in the 4 layers (not VIP1)


Cpl1 = np.loadtxt('import_files/Cpl1.txt') #connenctions probabilities from each of the 16 groups and VIP1
Csl1=np.loadtxt('import_files/Csl1.txt') #connenctions strenghts from each of the 16 groups and VIP1
Cp_tol1 = np.loadtxt('import_files/Cptol1.txt') #connenctions probabilities from VIP1 to each of the 16 groups
Cs_tol1 = np.loadtxt('import_files/Cstol1.txt') #connenctions strengths from VIP1 to each of the 16 groups
#print(Cp_tol1)
Cs_l1_l1=1.73 #connenctions strengths from VIP1 to VIP1
Cp_l1_l1=0.656 #connenctions probability from VIP1 to VIP1
# print(Cs)
# Cs[4*1+0][4*1+1]


#Parameters of the neurons for each layer
# row: layer in this order from top to bottom: 2_3,4,5,6
# column: populations in this order: e, pv, sst, vip
Cm=np.loadtxt('import_files/Cm.txt') #pF
gl=np.loadtxt('import_files/gl.txt') #nS
Vl=np.loadtxt('import_files/Vl.txt') #mV
Vr=np.loadtxt('import_files/Vr.txt') #mV
Vt=np.loadtxt('import_files/Vt.txt') #mV
tau_ref=np.loadtxt('import_files/tau_ref.txt') #ms


#Parameters of VIP1
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
print('--------------------------------------------------')

# In[10]:

#times for the inputs:
#when I want that an input that is on at t1 then off at t2 then on at t3 then off at t4:
t1=700
t2=700
t3=1000
t4=1300


# In[11]:

#Equations of the model. Each neuron is governed by this equations
eqs='''
        dv / dt = (- g_m * (v - V_L) - I_syn) / C_m : volt (unless refractory)
        I_syn = I_AMPA_rec + I_AMPA_ext + I_GABA + I_NMDA + I_external: amp

        #Parameters that can differ for each type of neuron, they are internal variable of the neuron.
        #This way I can then set their value later when I build the population. Each population can have a different value
        C_m : farad
        g_m: siemens
        V_L : volt
        V_rest : volt
        Vth: volt
        g_AMPA_ext: siemens
        g_AMPA_rec : siemens
        g_NMDA : siemens
        g_GABA :siemens

        #Here I define the external input, depending on what I want to study I can use one of this Equations. Just uncomment the one you want

        #If I want same input for the entire simulation use this:
        #I_external = I_ext: amp

        #When I want no input and then input activated use this:
        I_external= (abs(t-t1*ms)/(t-t1*ms) + 1)* (I_ext/2) : amp #at the beginnig is 0 then the input is activated

        #If I want: at the beginnig I is 0 then the input is activated then deactivated then activated again use this:
        #I_external= (abs(t-t1*ms)/(t-t1*ms) + 1) * (I_ext/2)-(abs(t-t2*ms)/(t-t2*ms) + 1) * (I_ext/2) + (abs(t-t3*ms)/(t-t3*ms) + 1) * (I_ext/2)- (abs(t-t4*ms)/(t-t4*ms) + 1) * (I_ext/2) : amp

        # #When I have 2 inputs at different times going to different layers use this:
        # #I need Iext and Iext1
        # I_external= (abs(t-t1*ms)/(t-t1*ms) + 1)* (I_ext/2) + (abs(t-t2*ms)/(t-t2*ms) + 1)* (I_ext1/2) : amp #at the beginnig is 0 then the input is activated

        #These are also variable of each neuron, I can later set the value I want when I build them

        I_ext : amp
        I_ext1 : amp #the second input to the other layer





        #Equations for AMPA receiving the inputs from the background (Poisson genetors:)

        I_AMPA_ext= g_AMPA_ext * (v - V_E) * w_ext * s_AMPA_ext : amp
        ds_AMPA_ext / dt = - s_AMPA_ext / tau_AMPA : 1
        #w_ext: 1 (If you want to have different weight fo each group, use this and uncomment later in pop.w_ext to set the desired value)
        # Here I don't need the summed variable because the neuron receive inputs from only one Poisson generator.
        #Each neuron need only one s.

        #Equations for AMPA receiving the inputs from other neurons:

        I_AMPA_rec = g_AMPA_rec * (v - V_E) * 1 * s_AMPA_tot : amp
        s_AMPA_tot=s_AMPA_tot0+s_AMPA_tot1+s_AMPA_tot2+s_AMPA_tot3 : 1
        s_AMPA_tot0 : 1
        s_AMPA_tot1 : 1
        s_AMPA_tot2 : 1
        s_AMPA_tot3 : 1
        #the eqs_ampa solve many s and sum them and give the summed value here
        #Each neuron receives inputs from many neurons. Each of them has his own differential equation s_AMPA (where I have the deltas with the spikes).
        #I then sum all the solutions s of the differential equations and I obtain s_AMPA_tot_post.
        #One s_AMPA_tot from each group of neurons sending excitation (each neuron is receiving from 4 groups)


        #Equations for GABA receiving the inputs from other neurons:


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

#This is the general ampa equation for each neuron type
eqs_ampa_base='''
            s_AMPA_tot_post= w_AMPA* s_AMPA : 1 (summed) #sum all the s, one for each synapse
            ds_AMPA / dt = - s_AMPA / tau_AMPA : 1 (clock-driven)
            w_AMPA: 1
        '''
#I need that each neuron group has his own AMPA equation,
# each group in fact has his own s_AMPA_tot, I create a list of equations
eqs_ampa=[]

for k in range (4):
    eqs_ampa.append(eqs_ampa_base.replace('s_AMPA_tot_post','s_AMPA_tot'+str(k)+'_post'))

#This is the general nmda equation for each neuron type
eqs_nmda_base='''s_NMDA_tot_post = w_NMDA * s_NMDA : 1 (summed)
    ds_NMDA / dt = - s_NMDA / tau_NMDA_decay + alpha_NMDA * x * (1 - s_NMDA) : 1 (clock-driven)
    dx / dt = - x / tau_NMDA_rise : 1 (clock-driven)
    w_NMDA : 1
'''
#I need that each neuron group has his own NMDA equation,
# each group in fact has his own s_NMDA_tot, I create a list of equations
eqs_nmda=[]
for k in range (4):
    eqs_nmda.append(eqs_nmda_base.replace('s_NMDA_tot_post','s_NMDA_tot'+str(k)+'_post'))

#This is the general gaba equation for each neuron type
eqs_gaba_base='''
    s_GABA_tot_post= w_GABA* s_GABA : 1 (summed)
    ds_GABA/ dt = - s_GABA/ tau_GABA : 1 (clock-driven)
    w_GABA: 1
'''
#I need that each neuron group has his own GABA equation,
# each group in fact has his own s_GABA_tot, I create a list of equations
eqs_gaba=[]
for k in range (12):
    eqs_gaba.append(eqs_gaba_base.replace('s_GABA_tot_post','s_GABA_tot'+str(k)+'_post'))

#Eqs I need to use for connections coming from L1, I only need GABA because VIP1 is inhibitory
eqs_gaba_l1= '''s_GABA_tot12_post= w_GABA* s_GABA : 1 (summed)
    ds_GABA/ dt = - s_GABA/ tau_GABA : 1 (clock-driven)
    w_GABA: 1
'''


# In[12]:
end_import= time.time() #To compute the time of import

# In[13]:

start_populations= time.time()
#def create_populations(N,eqs,Vt,Vr,Cm,gl,Vl,tau_ref,Iext): #at the end I don't create the function, I just do it
#I am creating all the populations in each layer
pops=[[],[],[],[]]
for h in range(0,4):
    for z in range(0,4):

        #Vth= Vt[h][z]*mV #The values are taken from the matrices with values
        #Vrest=Vr[h][z]*mV
        #Vrest=-78.85*mV

        pop = NeuronGroup(N[h][z], model=eqs, threshold='v > Vth', reset='v = V_rest', refractory=tau_ref[h][z]*ms, method='euler')

        pop.C_m = Cm[h][z]* pF
        pop.g_m= gl[h][z]*nS
        pop.V_L = Vl[h][z] *mV
        pop.V_I= Vl[h][z] *mV
        pop.V_rest= Vr[h][z] *mV
        pop.Vth=Vt[h][z]*mV
        #I am using the same value for everyone
        pop.g_AMPA_ext= 1*nS
        #pop.w_ext= 1 #wext[h][z] # I chose the same for everyone now
        pop.g_AMPA_rec = 1*nS #0.95*nS
        pop.g_NMDA = 1*nS #0.05*nS
        pop.g_GABA = 1*nS

        pop.I_ext= Iext[h][z]* pA
        pop.I_ext1= Iext1[h][z]* pA #If I want an input to another group


        #I initialize the starting value of the membrane potential
        for k in range(0,int(N[h][z])):
            pop[k].v[0]=Vr[h][z] *mV

        pops[h].append(pop)
        del (pop)
#return pops

#Here is where I can say which kinf of input I want
#I am giving the input to a subfraction of E4
pops[1][0][:int(N[1][0]/2)].I_ext=-30 * pA
#pops[0][0][:int(N[0][0]/2)].I_ext1=-30 * pA

#pops[1][0][:int(N[1][0]/3)].I_ext=-30 * pA
#pops[0][3][:].I_ext=+70 * pA
#pops[0][1][:].I_ext=+70 * pA


# def create_pop_l1(Nl1,Vt_l1,Vr_l1,Cm_l1,gl_l1,Vl_l1,tau_ref_l1,Iext_l1):  #at the end I don't create the function, I just do it
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


# In[14]:

#I create a poisson generator for each neuron in the population, all the neurons infact are receiving the inputs
#Function to connect each group to the noise
def input_layer_connect(Num,pop,gext,nu_ext): #nu_ext must be in Hz!!
    extinput=PoissonGroup(Num, rates = nu_ext)
    extconn = Synapses(extinput, pop, 'w: 1 ',on_pre='s_AMPA_ext += w')
    extconn.connect(j='i')
    extconn.w= gext #how much you affect s_ampa with 1 spike from the Poisson
    return extinput,extconn

# In[15]:

start_noise_conn= time.time()
#I call each time the function to connect each group to noise and save the connections (needed for Brian simulation)
print("Connecting noise devices to pupulations")

#nu_ext=np.loadtxt('import_files/nu_ext.txt') # Is at the beginning in the definitions of parameters!
#gext=1 is at the beginnig in the definitions of parameters!

all_extinput=[] #I have to save them in a list, I need to pass them in the "Network" function of Brian when you start simulations
all_extconn=[] #I have to save them in a list, I need to pass them in the "Network" function of Brian when you start simulations

#LAYER 2/3
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

#Delate, they are in the all_ext, they just occupy memory
del extinput_23e,extinput_23pv,extinput_23sst,extinput_23vip
del extconn_23e,extconn_23pv,extconn_23sst,extconn_23vip

#LAYER 4
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

#LAYER 5
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

#LAYER 6
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

#Connect L1 to noise
extinput_1,extconn_1=input_layer_connect(Nl1,popl1,gext,nu_extl1) #Connecting vipL1 to noise

all_extinput.append(extinput_1)
all_extconn.append(extconn_1)

del extinput_1,extconn_1
#I HAD TO CREATE LISTS TO SAVE EVERYTHING
#Brian needs all these included in Network before the simulation starts (see later)


# In[16]:
#IMPORTANT!!
#THIS IS THE STRUCTURE OF THE FOLLOWING VERY LONG FUNCTION:
#read here to understand

# def connect_populations(sources list, targets list, flag nmda, weights matrix, propbability matrix, N matrix, populations):
#             for loop on the sources
#                 for loop on the targets
#                     activate AMPA or GABA depending on the neuron type
#                     set the parameters

#                     activate NMDA

# In[17]:


#sources=[[layer,cell_type],[layer,cell_type]]  #sources[k][0] is the layer
                                                #sources[k][1] is the cell type


# In[18]:
def connect_populations(sources,targets,G,Cs,Cp,N,pops,d,e_ampa,e_nmda,i_ampa,i_nmda,nmda_on=True):

    #percentage of different receptors, I multiply the prob of connection by this. (now is passed in the function no need to have it here)
#     e_ampa=0.8
#     e_nmda=0.2
#     i_ampa=0.8
#     i_nmda=0.2

    All_C=[] #I store all the connections here

    #In the future I can have connections within same population stronger/weaker than the one between different populations
    wp_p=1  #multyply factor for connections within the same populations
    wp_m=1  #multyply factor for connections between different populations

    for h in range(len(sources)):
        for k in range(len(targets)):
            s_layer = sources[h][0] #sending layer
            s_cell_type = sources[h][1] #population type in the sending layer
            t_layer = targets[k][0] #target layer
            t_cell_type = targets[k][1] #population type in the target layer

            if s_cell_type==0: # sendind is excitatory neuron

                if t_cell_type==0: # target is excitatory neuron

                    #sending is excitaotry receiving is excitaotry then they are connected trought AMPA receptors:
                    conn= Synapses(pops[s_layer][s_cell_type],pops[t_layer][t_cell_type],model=eqs_ampa[s_layer],on_pre='s_AMPA+=1', method='euler')
                    conn.connect(condition='i != j',p=e_ampa*Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type])
                    #conn.connect(condition='i != j',p=Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*1)# when NMDA off use this

                    if s_layer==t_layer and s_cell_type==t_cell_type: #within the same population
                        wp=wp_p
                    else:  #between different populations
                        wp=wp_m
                    #print("Printing the connections")
                    #print(conn.N_outgoing_pre)
                    if Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]==0: #If the probability of connections is 0 I still need to create an AMPA that has a weight with 0 effect
                        conn.w_AMPA= 0
                    else:
                        conn.w_AMPA=wp* G*Cs[4*s_layer+s_cell_type][4*t_layer+t_cell_type]/(e_ampa*Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*N[s_layer][s_cell_type])

                    conn.delay='d'
                    All_C.append(conn) #I append the connections to the list containing all of them
                    del conn #I delete it to save memory

                    if nmda_on==True: #I need to create the NMDA connections
                        conn1= Synapses(pops[s_layer][s_cell_type],pops[t_layer][t_cell_type],model=eqs_nmda[s_layer],on_pre='x+=1', method='euler')
                        conn1.connect(condition='i != j',p=e_nmda*Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type])

                        if Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]==0: #If the probability of connections is 0 I still need to create an NMDA that has a weight with 0 effect
                            conn1.w_NMDA= 0
                        else:
                            conn1.w_NMDA=wp* G* Cs[4*s_layer+s_cell_type][4*t_layer+t_cell_type]/(e_nmda*Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*N[s_layer][s_cell_type])

                        conn1.delay='d'
                        All_C.append(conn1)
                        del conn1

                if t_cell_type!=0: # target is inhibitory neuron
                                    #(Note: is the same as before but in the future if the % of AMPA is different I have already everything in place)

                    conn= Synapses(pops[s_layer][s_cell_type],pops[t_layer][t_cell_type],model=eqs_ampa[s_layer],on_pre='s_AMPA+=1', method='euler')
                    conn.connect(condition='i != j',p=i_ampa*Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type])
                    #conn.connect(condition='i != j',p=Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*1)# when NMDA off use this

                    if s_layer==t_layer and s_cell_type==t_cell_type: #within the same population
                        wp=wp_p
                    else: #between different populations
                        wp=wp_m
                    #print("Printing the connections")
                    #print(conn.N_outgoing_pre)
                    if Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]==0: #If the probability of connections is 0 I still need to create an AMPA that has a weight with 0 effect
                        conn.w_AMPA= 0
                    else:
                        conn.w_AMPA=wp*G*Cs[4*s_layer+s_cell_type][4*t_layer+t_cell_type]/(i_ampa*Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*N[s_layer][s_cell_type])

                    conn.delay='d'
                    All_C.append(conn) #I append the connections to the list containing all of them
                    del conn #I delete it to save memory

                    if nmda_on==True: #I need to create the NMDA connections
                        conn1= Synapses(pops[s_layer][s_cell_type],pops[t_layer][t_cell_type],model=eqs_nmda[s_layer],on_pre='x+=1', method='euler')
                        conn1.connect(condition='i != j',p=i_nmda*Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type])
                        #conn1.connect(condition='i != j',p=Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*1) # when I try weights instead
                        if Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]==0: #If the probability of connections is 0 I still need to create an AMPA that has a weight with 0 effect
                            conn1.w_NMDA= 0
                        else:
                            conn1.w_NMDA=wp*G* Cs[4*s_layer+s_cell_type][4*t_layer+t_cell_type]/(i_nmda*Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*N[s_layer][s_cell_type])

                        conn1.delay='d'
                        All_C.append(conn1)
                        del conn1

            else: # sendind is inhibitory neuron, the connections goes to GABA receptors
                conn2= Synapses(pops[s_layer][s_cell_type],pops[t_layer][t_cell_type],model=eqs_gaba[3*s_layer+s_cell_type-1],on_pre='s_GABA+=1', method='euler')
                conn2.connect(condition='i != j',p=Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type])

                if s_layer==t_layer and s_cell_type==t_cell_type: #within the same population
                    wp=wp_p
                else: #between different populations
                    wp=wp_m

                if Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]==0:  #If the probability of connections is 0 I still need to create an GABA that has a weight with 0 effect
                    conn2.w_GABA= 0
                else:
                    conn2.w_GABA=wp*G* Cs[4*s_layer+s_cell_type][4*t_layer+t_cell_type]/(Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*N[s_layer][s_cell_type])
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
#IMPORTANT to understand:
# I have to assign to each source his own equation bijectively. This eqs_gaba[3*s_layer+s_cell_type-1]
#trasform the pair [layer][cell_type] into a number corresponding to one of the 11 gaba equations


# I want a correspondace between my matrix 4x4 (layer,cell type) and the matrix 16x16 where all the values of the connections are stored.
# This is why I need #Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]



# In[19]:

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
                conn2.w_GABA=Gl1* Csl1[4*t_layer+t_cell_type]/(Cpl1[4*t_layer+t_cell_type]*Nl1)
            conn2.delay='d'
            All_C_l1.append(conn2)
            del conn2
    return All_C_l1


# In[20]:

#Function to connect l1 to l1
def connect_l1_l1(Gl1,Cs_l1_l1,Cp_l1_l1,Nl1,popl1,d):
    conn2= Synapses(popl1,popl1,model=eqs_gaba_l1,on_pre='s_GABA+=1', method='euler')
    conn2.connect(condition='i != j',p=Cp_l1_l1)
    #conn2.w_GABA= Cs_l1_l1
    conn2.w_GABA= Gl1* Cs_l1_l1/(Cp_l1_l1*Nl1)
    conn2.delay='d'
    return conn2


# In[21]:

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
            if Cs_tol1[4*s_layer+s_cell_type]==0 or Cp_tol1[4*s_layer+s_cell_type]==0:
                conn.w_AMPA= 0
            else:
                conn.w_AMPA=Gl1* Cs_tol1[4*s_layer+s_cell_type]/(i_ampa*Cp_tol1[4*s_layer+s_cell_type]*N[s_layer][s_cell_type])

            conn.delay='d'
            All_C.append(conn)
            del conn

            if nmda_on==True:
                conn1= Synapses(pops[s_layer][s_cell_type],popl1,model=eqs_nmda[s_layer],on_pre='x+=1', method='euler')
                conn1.connect(condition='i != j',p=Cp_tol1[4*s_layer+s_cell_type]*i_nmda)
                if Cs_tol1[4*s_layer+s_cell_type]==0 or Cp_tol1[4*s_layer+s_cell_type]==0:
                    conn1.w_NMDA= 0
                else:
                    conn1.w_NMDA=Gl1*  Cs_tol1[4*s_layer+s_cell_type]/(i_nmda*Cp_tol1[4*s_layer+s_cell_type]*N[s_layer][s_cell_type])

                conn1.delay='d'
                All_C.append(conn1)
                del conn1
        else:
            conn2= Synapses(pops[s_layer][s_cell_type],popl1,model=eqs_gaba[3*s_layer+s_cell_type-1],on_pre='s_GABA+=1', method='euler')
            conn2.connect(condition='i != j',p=Cp_tol1[4*s_layer+s_cell_type])

            if Cs_tol1[4*s_layer+s_cell_type]==0 or Cp_tol1[4*s_layer+s_cell_type]==0:
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


# In[22]:
#DIFFERENT FUNCIONS TO CONNECT LAYERS

#CONNECTING ALL LAYERS (layer 2/3, 4, 5, 6)
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

#CONNECTING only 2 LAYERS
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

#connection L1 to all & all to L1
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

# In[23]:

#SIMPLE TESTS
# sources=[[0,1],[0,2],[0,3]]
# targets=[[0,1],[0,2],[0,3]]
# con_test=connect_populations(sources,targets,G Cs,Cp,N,pops,d,nmda_on=True) #e_ampa to insert!
# #print(con_test)
# connections=con_test


# In[24]:

#conn4_4=connect_layers(1,1,G,Cs,Cp,N,pops,d,nmda_on=True)
# conn23_23=connect_layers(0,0,Cs,Cp,G,N,pops,d,nmda_on=True)
# conn23to4=connect_layers(0,1,Cs,Cp,G,N,pops,d,nmda_on=True)
# connections= conn23_23 + conn23to4


# In[25]:

#I connect all the layers by calling the functions to connect
start_connecting=time.time()
print('--------------------------------------------------')
print('Connecting layers')
conn_all_l1=connect_l1_all(Gl1,Csl1,Cpl1,Cs_tol1,Cp_tol1,Cs_l1_l1,Cp_l1_l1,N,Nl1,pops,popl1,d,i_ampa,i_nmda,nmda_on=True)
conn_all=connect_all_layers(Cs,Cp,G,N,pops,d,e_ampa,e_nmda,i_ampa,i_nmda,nmda_on=True)
connections=conn_all+conn_all_l1 #all the connections are saved in this array
print('All layers now connected')
end_connecting=time.time()


# In[26]:

#FUNCTIONS TO MONITOR
#spike detectors
def spike_det(pops,layer,rec=True):
    e_spikes = SpikeMonitor(pops[layer][0],record=rec) #create the spike detector for e
    pv_spikes= SpikeMonitor(pops[layer][1],record=rec) #create the spike detector for subgroup pv
    sst_spikes= SpikeMonitor(pops[layer][2],record=rec) #create the spike detector for subgroup sst
    vip_spikes= SpikeMonitor(pops[layer][3],record=rec)#create the spike detector for subgroup vip

    return e_spikes,pv_spikes,sst_spikes,vip_spikes

#subgroup where from each group I record a number n_activity
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


# In[27]:
#------------------------------------------------------------------------------
# Create the recording devices: Spike detectors and rate detectors (calling the functions)
#------------------------------------------------------------------------------

start_detectors=time.time()

#mE=StateMonitor(pops[0][0][10], 'v',record=True) #check membrane potential of one neuron
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



# In[28]:


#FOR LFP RECORDINGS I NEED THESE
#num=np.array(np.loadtxt('LFP_files/numberLFP.txt') ) #Array containing the number of neuron in each sub section of the sphere of the electrode
#print(num)

def LFP_recording(num,pops):
    indeces23=[i for i in range(0,int(num[0]))]
    indeces4=[i for i in range(0,int(num[1]))]
    indeces5=[i for i in range(0,int(num[2]))]
    indeces6=[i for i in range(0,int(num[3]))]

    #I only record the subfractions of neurons which are inside the sphere of resolution of the electrode, info contained in num
    iampaE23=StateMonitor(pops[0][0], 'I_AMPA_rec',record=indeces23)
    igabaE23=StateMonitor(pops[0][0], 'I_GABA',record=indeces23)
    inmdaE23=StateMonitor(pops[0][0], 'I_NMDA',record=indeces23)
    iampaextE23=StateMonitor(pops[0][0], 'I_AMPA_ext',record=indeces23)

    iampaE4=StateMonitor(pops[1][0], 'I_AMPA_rec',record=indeces4)
    igabaE4=StateMonitor(pops[1][0], 'I_GABA',record=indeces4)
    inmdaE4=StateMonitor(pops[1][0], 'I_NMDA',record=indeces4)
    iampaextE4=StateMonitor(pops[1][0], 'I_AMPA_ext',record=indeces4)

    iampaE5=StateMonitor(pops[2][0], 'I_AMPA_rec',record=indeces5)
    igabaE5=StateMonitor(pops[2][0], 'I_GABA',record=indeces5)
    inmdaE5=StateMonitor(pops[2][0], 'I_NMDA',record=indeces5)
    iampaextE5=StateMonitor(pops[2][0], 'I_AMPA_ext',record=indeces5)

    iampaE6=StateMonitor(pops[3][0], 'I_AMPA_rec',record=indeces6)
    igabaE6=StateMonitor(pops[3][0], 'I_GABA',record=indeces6)
    inmdaE6=StateMonitor(pops[3][0], 'I_NMDA',record=indeces6)
    iampaextE6=StateMonitor(pops[3][0], 'I_AMPA_ext',record=indeces6)

    return iampaE23,igabaE23,inmdaE23,iampaextE23,iampaE4,igabaE4,inmdaE4,iampaextE4,iampaE5,igabaE5,inmdaE5,iampaextE5,iampaE6,igabaE6,inmdaE6,iampaextE6

#iampaE23,igabaE23,inmdaE23,iampaextE23,iampaE4,igabaE4,inmdaE4,iampaextE4,iampaE5,igabaE5,inmdaE5,iampaextE5,iampaE6,igabaE6,inmdaE6,iampaextE6=LFP_recording(num,pops)


# In[29]:
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

#------------------------------------------------------------------------------
# Run the simulation
#------------------------------------------------------------------------------

defaultclock.dt = dt_sim*ms #time step of simulations
#runtime = 800.0 * ms  # total simulation (moved at time at the biginnig of the program)

# construct network
#I have to add here all the populations, inputs, connections, monitor devices
net = Network(pops[:],popl1,all_extinput[:],all_extconn[:],
              connections[:],
              #mE,
              S_vip1,R_vip1,
              S_e4,S_pv4,S_sst4,S_vip4,
              S_e5,S_pv5,S_sst5,S_vip5,
              S_e6,S_pv6,S_sst6,S_vip6,
              S_e23,S_pv23,S_sst23,S_vip23,
             R_e23,R_pv23,R_sst23,R_vip23,
             R_e4,R_pv4,R_sst4,R_vip4,
             R_e5,R_pv5,R_sst5,R_vip5,
             R_e6,R_pv6,R_sst6,R_vip6
#               iampaE23,inmdaE23,igabaE23,iampaextE23,
#              iampaE4,inmdaE4,igabaE4,iampaextE4,
#              iampaE5,inmdaE5,igabaE5,iampaextE5,
#              iampaE6,inmdaE6,igabaE6,iampaextE6
             )

print('Network is Built')
endbuild = time.time()

# THIS IS TO SAVE THE CONNECIONS WEIGHTS (the indexes corresponfing to the group can be found in my handnotes)
# outfile='W_88_in'
# np.save(outfile, connections[88].w_AMPA)
#
#
# outfile='W_0_in'
# np.save(outfile, connections[0].w_AMPA)
#
# outfile='W_8_in'
# np.save(outfile, connections[8].w_AMPA)
#
#
# outfile='W_16_in'
# np.save(outfile, connections[16].w_AMPA)
#
# outfile='W_80_in'
# np.save(outfile, connections[80].w_AMPA)
#
# outfile='W_96_in'
# np.save(outfile, connections[96].w_AMPA)
#
# outfile='W_104_in'
# np.save(outfile, connections[104].w_AMPA)
#
# outfile='W_160_in'
# np.save(outfile, connections[160].w_AMPA)
#
# outfile='W_168_in'
# np.save(outfile, connections[168].w_AMPA)
#
# outfile='W_176_in'
# np.save(outfile, connections[176].w_AMPA)
#
# outfile='W_184_in'
# np.save(outfile, connections[184].w_AMPA)
#
# outfile='W_256_in'
# np.save(outfile, connections[256].w_AMPA)
#
# outfile='W_264_in'
# np.save(outfile, connections[264].w_AMPA)

#START THE SIMULATION
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


# In[30]:

#print(igabaE23.I_GABA[0])

# In[31]:
#I save the currents AMPA,GABA,NMDA of the neurons in files, so then I can compute the LFP
def save_lfp_rec():
    #layer 2/3
    igabaE23_save=[]
    iampaE23_save=[]
    inmdaE23_save=[]
    iampaextE23_save=[]

    for i in range(0,int(num[0])): #num[0] is the number of neurons I recorded from layer 2/3
        igabaE23_save.append(igabaE23.I_GABA[i]/pA)
        iampaE23_save.append(iampaE23.I_AMPA_rec[i]/pA)
        inmdaE23_save.append(inmdaE23.I_NMDA[i]/pA)
        iampaextE23_save.append(iampaextE23.I_AMPA_ext[i]/pA)

    igabaE23_save=np.array(igabaE23_save)
    iampaE23_save=np.array(iampaE23_save)
    inmdaE23_save=np.array(inmdaE23_save)
    iampaextE23_save=np.array(iampaextE23_save)

    #print(len(igabaE23_save)) #is the number of neurons I recorded from
    #print(len(igabaE23_save[0])) #is the number of steps of the sim, currents over time of neuron 0

    np.save("LFP_files/igabaE23.npy", igabaE23_save)
    np.save("LFP_files/iampaE23.npy", iampaE23_save)
    np.save("LFP_files/inmdaE23.npy", inmdaE23_save)
    np.save("LFP_files/iampaextE23.npy", iampaextE23_save)

    #layer 4
    igabaE4_save=[]
    iampaE4_save=[]
    inmdaE4_save=[]
    iampaextE4_save=[]
    for i in range(0,int(num[1])):
        igabaE4_save.append(igabaE4.I_GABA[i]/pA)
        iampaE4_save.append(iampaE4.I_AMPA_rec[i]/pA)
        inmdaE4_save.append(inmdaE4.I_NMDA[i]/pA)
        iampaextE4_save.append(iampaextE4.I_AMPA_ext[i]/pA)

    igabaE4_save=np.array(igabaE4_save)
    iampaE4_save=np.array(iampaE4_save)
    inmdaE4_save=np.array(inmdaE4_save)
    iampaextE4_save=np.array(iampaextE4_save)

    np.save("LFP_files/igabaE4.npy", igabaE4_save)
    np.save("LFP_files/iampaE4.npy", iampaE4_save)
    np.save("LFP_files/inmdaE4.npy", inmdaE4_save)
    np.save("LFP_files/iampaextE4.npy", iampaextE4_save)

    #layer 5
    igabaE5_save=[]
    iampaE5_save=[]
    inmdaE5_save=[]
    iampaextE5_save=[]
    for i in range(0,int(num[2])):
        igabaE5_save.append(igabaE5.I_GABA[i]/pA)
        iampaE5_save.append(iampaE5.I_AMPA_rec[i]/pA)
        inmdaE5_save.append(inmdaE5.I_NMDA[i]/pA)
        iampaextE5_save.append(iampaextE5.I_AMPA_ext[i]/pA)

    igabaE5_save=np.array(igabaE5_save)
    iampaE5_save=np.array(iampaE5_save)
    inmdaE5_save=np.array(inmdaE5_save)
    iampaextE5_save=np.array(iampaextE5_save)

    np.save("LFP_files/igabaE5.npy", igabaE5_save)
    np.save("LFP_files/iampaE5.npy", iampaE5_save)
    np.save("LFP_files/inmdaE5.npy", inmdaE5_save)
    np.save("LFP_files/iampaextE5.npy", iampaextE5_save)

    #layer 6
    igabaE6_save=[]
    iampaE6_save=[]
    inmdaE6_save=[]
    iampaextE6_save=[]
    for i in range(0,int(num[3])):
        igabaE6_save.append(igabaE6.I_GABA[i]/pA)
        iampaE6_save.append(iampaE6.I_AMPA_rec[i]/pA)
        inmdaE6_save.append(inmdaE6.I_NMDA[i]/pA)
        iampaextE6_save.append(iampaextE6.I_AMPA_ext[i]/pA)

    igabaE6_save=np.array(igabaE6_save)
    iampaE6_save=np.array(iampaE6_save)
    inmdaE6_save=np.array(inmdaE6_save)
    iampaextE6_save=np.array(iampaextE6_save)

    np.save("LFP_files/igabaE6.npy", igabaE6_save)
    np.save("LFP_files/iampaE6.npy", iampaE6_save)
    np.save("LFP_files/inmdaE6.npy", inmdaE6_save)
    np.save("LFP_files/iampaextE6.npy", iampaextE6_save)
#save_lfp_rec()


# In[32]:

#print(igabaE23.I_GABA[1])

# In[33]:

# print(len(igabaE4.I_GABA))
# print(num[1])

# In[34]:

# igabaE23test=np.load("LFP_files/igabaE23.npy")
# print(igabaE23test[1])

# In[35]:

# fig2 = plt.figure(figsize=(15,7))
# plot(mE.t/ms,mE.v[0],label='e')
# xlabel('time (ms)')
# ylabel('Membran potential V (mV)')
# legend()
# show()



# In[37]:

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
f=open(a+"/S_vip1numspike.txt",'w+') #create the file
f.write('%f ' %S_vip1.num_spikes)
f.close()

#layer 2/3
f=open(a+"/S_e23numspike.txt",'w+') #create the file
f.write('%f ' %S_e23.num_spikes)
f.close()

f=open(a+"/S_pv23numspike.txt",'w+') #create the file
f.write('%f ' %S_pv23.num_spikes)
f.close()

f=open(a+"/S_sst23numspike.txt",'w+') #create the file
f.write('%f ' %S_sst23.num_spikes)
f.close()

f=open(a+"/S_vip23numspike.txt",'w+') #create the file
f.write('%f ' %S_vip23.num_spikes)
f.close()

#layer4
f=open(a+"/S_e4numspike.txt",'w+') #create the file
f.write('%f ' %S_e4.num_spikes)
f.close()

f=open(a+"/S_pv4numspike.txt",'w+') #create the file
f.write('%f ' %S_pv4.num_spikes)
f.close()

f=open(a+"/S_sst4numspike.txt",'w+') #create the file
f.write('%f ' %S_sst4.num_spikes)
f.close()

f=open(a+"/S_vip4numspike.txt",'w+') #create the file
f.write('%f ' %S_vip4.num_spikes)
f.close()

#layer 5
f=open(a+"/S_e5numspike.txt",'w+') #create the file
f.write('%f ' %S_e5.num_spikes)
f.close()

f=open(a+"/S_pv5numspike.txt",'w+') #create the file
f.write('%f ' %S_pv5.num_spikes)
f.close()

f=open(a+"/S_sst5numspike.txt",'w+') #create the file
f.write('%f ' %S_sst5.num_spikes)
f.close()

f=open(a+"/S_vip5numspike.txt",'w+') #create the file
f.write('%f ' %S_vip5.num_spikes)
f.close()

#layer6
f=open(a+"/S_e6numspike.txt",'w+') #create the file
f.write('%f ' %S_e6.num_spikes)
f.close()

f=open(a+"/S_pv6numspike.txt",'w+') #create the file
f.write('%f ' %S_pv6.num_spikes)
f.close()

f=open(a+"/S_sst6numspike.txt",'w+') #create the file
f.write('%f ' %S_sst6.num_spikes)
f.close()

f=open(a+"/S_vip6numspike.txt",'w+') #create the file
f.write('%f ' %S_vip6.num_spikes)
f.close()


# In[42]:

#I write the spikes in files of every neuron. In one file there are the indeces
#.i of th neurons emitting spike at the corrisponding time of the other file .t

#layer 1
f=open(a+"/S_vip1i.txt",'w+') #create the file
for i in range(0,len(S_vip1.i)):
    f.write('%i ' %S_vip1.i[i])
    f.write('\n')
f.close()

f=open(a+"/S_vip1t.txt",'w+') #create the file
for i in range(0,len(S_vip1.t)):
    f.write('%f ' %S_vip1.t[i])
    f.write('\n')
f.close()

#layer 2/3
f=open(a+"/S_e23i.txt",'w+') #create the file
for i in range(0,len(S_e23.i)):
    f.write('%i ' %S_e23.i[i])
    f.write('\n')
f.close()

f=open(a+"/S_e23t.txt",'w+') #create the file
for i in range(0,len(S_e23.t)):
    f.write('%f ' %S_e23.t[i])
    f.write('\n')
f.close()

f=open(a+"/S_pv23i.txt",'w+') #create the file
for i in range(0,len(S_pv23.i)):
    f.write('%i ' %S_pv23.i[i])
    f.write('\n')
f.close()

f=open(a+"/S_pv23t.txt",'w+') #create the file
for i in range(0,len(S_pv23.t)):
    f.write('%f ' %S_pv23.t[i])
    f.write('\n')
f.close()


f=open(a+"/S_sst23i.txt",'w+') #create the file
for i in range(0,len(S_sst23.i)):
    f.write('%i ' %S_sst23.i[i])
    f.write('\n')
f.close()

f=open(a+"/S_sst23t.txt",'w+') #create the file
for i in range(0,len(S_sst23.t)):
    f.write('%f ' %S_sst23.t[i])
    f.write('\n')
f.close()

f=open(a+"/S_vip23i.txt",'w+') #create the file
for i in range(0,len(S_vip23.i)):
    f.write('%i ' %S_vip23.i[i])
    f.write('\n')
f.close()

f=open(a+"/S_vip23t.txt",'w+') #create the file
for i in range(0,len(S_vip23.t)):
    f.write('%f ' %S_vip23.t[i])
    f.write('\n')
f.close()

#layer4
f=open(a+"/S_e4i.txt",'w+') #create the file
for i in range(0,len(S_e4.i)):
    f.write('%i ' %S_e4.i[i])
    f.write('\n')
f.close()

f=open(a+"/S_e4t.txt",'w+') #create the file
for i in range(0,len(S_e4.t)):
    f.write('%f ' %S_e4.t[i])
    f.write('\n')
f.close()

f=open(a+"/S_pv4i.txt",'w+') #create the file
for i in range(0,len(S_pv4.i)):
    f.write('%i ' %S_pv4.i[i])
    f.write('\n')
f.close()

f=open(a+"/S_pv4t.txt",'w+') #create the file
for i in range(0,len(S_pv4.t)):
    f.write('%f ' %S_pv4.t[i])
    f.write('\n')
f.close()

f=open(a+"/S_sst4i.txt",'w+') #create the file
for i in range(0,len(S_sst4.i)):
    f.write('%i ' %S_sst4.i[i])
    f.write('\n')
f.close()

f=open(a+"/S_sst4t.txt",'w+') #create the file
for i in range(0,len(S_sst4.t)):
    f.write('%f ' %S_sst4.t[i])
    f.write('\n')
f.close()


f=open(a+"/S_vip4i.txt",'w+') #create the file
for i in range(0,len(S_vip4.i)):
    f.write('%i ' %S_vip4.i[i])
    f.write('\n')
f.close()

f=open(a+"/S_vip4t.txt",'w+') #create the file
for i in range(0,len(S_vip4.t)):
    f.write('%f ' %S_vip4.t[i])
    f.write('\n')
f.close()

#layer 5
f=open(a+"/S_e5i.txt",'w+') #create the file
for i in range(0,len(S_e5.i)):
    f.write('%i ' %S_e5.i[i])
    f.write('\n')
f.close()

f=open(a+"/S_e5t.txt",'w+') #create the file
for i in range(0,len(S_e5.t)):
    f.write('%f ' %S_e5.t[i])
    f.write('\n')
f.close()

f=open(a+"/S_pv5i.txt",'w+') #create the file
for i in range(0,len(S_pv5.i)):
    f.write('%i ' %S_pv5.i[i])
    f.write('\n')
f.close()

f=open(a+"/S_pv5t.txt",'w+') #create the file
for i in range(0,len(S_pv5.t)):
    f.write('%f ' %S_pv5.t[i])
    f.write('\n')
f.close()

f=open(a+"/S_sst5i.txt",'w+') #create the file
for i in range(0,len(S_sst5.i)):
    f.write('%i ' %S_sst5.i[i])
    f.write('\n')
f.close()

f=open(a+"/S_sst5t.txt",'w+') #create the file
for i in range(0,len(S_sst5.t)):
    f.write('%f ' %S_sst5.t[i])
    f.write('\n')
f.close()


f=open(a+"/S_vip5i.txt",'w+') #create the file
for i in range(0,len(S_vip5.i)):
    f.write('%i ' %S_vip5.i[i])
    f.write('\n')
f.close()

f=open(a+"/S_vip5t.txt",'w+') #create the file
for i in range(0,len(S_vip5.t)):
    f.write('%f ' %S_vip5.t[i])
    f.write('\n')
f.close()


#layer6
f=open(a+"/S_e6i.txt",'w+') #create the file
for i in range(0,len(S_e6.i)):
    f.write('%i ' %S_e6.i[i])
    f.write('\n')
f.close()

f=open(a+"/S_e6t.txt",'w+') #create the file
for i in range(0,len(S_e6.t)):
    f.write('%f ' %S_e6.t[i])
    f.write('\n')
f.close()

f=open(a+"/S_pv6i.txt",'w+') #create the file
for i in range(0,len(S_pv6.i)):
    f.write('%i ' %S_pv6.i[i])
    f.write('\n')
f.close()

f=open(a+"/S_pv6t.txt",'w+') #create the file
for i in range(0,len(S_pv6.t)):
    f.write('%f ' %S_pv6.t[i])
    f.write('\n')
f.close()

f=open(a+"/S_sst6i.txt",'w+') #create the file
for i in range(0,len(S_sst6.i)):
    f.write('%i ' %S_sst6.i[i])
    f.write('\n')
f.close()

f=open(a+"/S_sst6t.txt",'w+') #create the file
for i in range(0,len(S_sst6.t)):
    f.write('%f ' %S_sst6.t[i])
    f.write('\n')
f.close()


f=open(a+"/S_vip6i.txt",'w+') #create the file
for i in range(0,len(S_vip6.i)):
    f.write('%i ' %S_vip6.i[i])
    f.write('\n')
f.close()

f=open(a+"/S_vip6t.txt",'w+') #create the file
for i in range(0,len(S_vip6.t)):
    f.write('%f ' %S_vip6.t[i])
    f.write('\n')
f.close()

#SAVE SPIKE COUNTS
# np.savetxt(a+'/spike_counts_vip1.txt', S_vip1.count, fmt='%d')
# np.savetxt(a+'/spike_counts_e23.txt', S_e23.count, fmt='%d')
# np.savetxt(a+'/spike_counts_pv23.txt', S_pv23.count, fmt='%d')
# np.savetxt(a+'/spike_counts_sst23.txt', S_sst23.count, fmt='%d')
# np.savetxt(a+'/spike_counts_vip23.txt', S_vip23.count, fmt='%d')
#
# np.savetxt(a+'/spike_counts_e4.txt', S_e4.count, fmt='%d')
# np.savetxt(a+'/spike_counts_pv4.txt', S_pv4.count, fmt='%d')
# np.savetxt(a+'/spike_counts_sst4.txt', S_sst4.count, fmt='%d')
# np.savetxt(a+'/spike_counts_vip4.txt', S_vip4.count, fmt='%d')
#
# np.savetxt(a+'/spike_counts_e5.txt', S_e5.count, fmt='%d')
# np.savetxt(a+'/spike_counts_pv5.txt', S_pv5.count, fmt='%d')
# np.savetxt(a+'/spike_counts_sst5.txt', S_sst5.count, fmt='%d')
# np.savetxt(a+'/spike_counts_vip5.txt', S_vip5.count, fmt='%d')
#
# np.savetxt(a+'/spike_counts_e6.txt', S_e6.count, fmt='%d')
# np.savetxt(a+'/spike_counts_pv6.txt', S_pv6.count, fmt='%d')
# np.savetxt(a+'/spike_counts_sst6.txt', S_sst6.count, fmt='%d')
# np.savetxt(a+'/spike_counts_vip6.txt', S_vip6.count, fmt='%d')

#
# SAVE SPIKE TRAINS
# sp_train_vip1 = list(S_vip1.spike_trains().values())
# np.save(a+'/sp_train_vip1.npy', sp_train_vip1, allow_pickle=True)
#
# sp_train_e23 = list(S_e23.spike_trains().values())
# np.save(a+'/sp_train_e23.npy', sp_train_e23, allow_pickle=True)
# sp_train_pv23 = list(S_pv23.spike_trains().values())
# np.save(a+'/sp_train_pv23.npy', sp_train_pv23, allow_pickle=True)
# sp_train_sst23 = list(S_sst23.spike_trains().values())
# np.save(a+'/sp_train_sst23.npy', sp_train_sst23, allow_pickle=True)
# sp_train_vip23 = list(S_vip23.spike_trains().values())
# np.save(a+'/sp_train_vip23.npy', sp_train_vip23, allow_pickle=True)
#
# sp_train_e4 = list(S_e4.spike_trains().values())
# np.save(a+'/sp_train_e4.npy', sp_train_e4, allow_pickle=True)
# sp_train_pv4 = list(S_pv4.spike_trains().values())
# np.save(a+'/sp_train_pv4.npy', sp_train_pv4, allow_pickle=True)
# sp_train_sst4 = list(S_sst4.spike_trains().values())
# np.save(a+'/sp_train_sst4.npy', sp_train_sst4, allow_pickle=True)
# sp_train_vip4 = list(S_vip4.spike_trains().values())
# np.save(a+'/sp_train_vip4.npy', sp_train_vip4, allow_pickle=True)
#
# sp_train_e5 = list(S_e5.spike_trains().values())
# np.save(a+'/sp_train_e5.npy', sp_train_e5, allow_pickle=True)
# sp_train_pv5 = list(S_pv5.spike_trains().values())
# np.save(a+'/sp_train_pv5.npy', sp_train_pv5, allow_pickle=True)
# sp_train_sst5 = list(S_sst5.spike_trains().values())
# np.save(a+'/sp_train_sst5.npy', sp_train_sst5, allow_pickle=True)
# sp_train_vip5 = list(S_vip5.spike_trains().values())
# np.save(a+'/sp_train_vip5.npy', sp_train_vip5, allow_pickle=True)
#
# sp_train_e6 = list(S_e6.spike_trains().values())
# np.save(a+'/sp_train_e6.npy', sp_train_e6, allow_pickle=True)
# sp_train_pv6 = list(S_pv6.spike_trains().values())
# np.save(a+'/sp_train_pv6.npy', sp_train_pv6, allow_pickle=True)
# sp_train_sst6 = list(S_sst6.spike_trains().values())
# np.save(a+'/sp_train_sst6.npy', sp_train_sst6, allow_pickle=True)
# sp_train_vip6 = list(S_vip6.spike_trains().values())
# np.save(a+'/sp_train_vip6.npy', sp_train_vip6, allow_pickle=True)
#

#Function to compute inter spike time interval
def ISI(N,time_spikes):

    ISI=[[] for i in range(N)]
    for n in range(0,N): # for each neuron
        for i in range(0,len(time_spikes[n])-1): # in the list of that neuron I do the difference
            ISI[n].append(time_spikes[n][i+1]-time_spikes[n][i]) # I have the difference between two spike for each pair of spike of that particular neuron
    return ISI

#COMPUTE IRREGULARITY FIG2 paper:
# ISI_e23=ISI(N[0][0],S_e23.spike_trains())
# ISI_e23 = [x for x in ISI_e23 if x != []]
# cvs_e23 = [ np.std(i)/np.mean(i) for i in ISI_e23]
# np.savetxt(a+'/cvs_e23.txt', cvs_e23)
# ISI_pv23=ISI(N[0][1],S_pv23.spike_trains())
# ISI_pv23 = [x for x in ISI_pv23 if x != []]
# cvs_pv23 = [ np.std(i)/np.mean(i) for i in ISI_pv23]
# np.savetxt(a+'/cvs_pv23.txt', cvs_pv23)
# ISI_sst23=ISI(N[0][2],S_sst23.spike_trains())
# ISI_sst23 = [x for x in ISI_sst23 if x != []]
# cvs_sst23 = [ np.std(i)/np.mean(i) for i in ISI_sst23]
# np.savetxt(a+'/cvs_sst23.txt', cvs_sst23)
# ISI_vip23=ISI(N[0][3],S_vip23.spike_trains())
# ISI_vip23 = [x for x in ISI_vip23 if x != []]
# cvs_vip23 = [ np.std(i)/np.mean(i) for i in ISI_vip23]
# np.savetxt(a+'/cvs_vip23.txt', cvs_vip23)

#
# ISI_e4=ISI(N[1][0],S_e4.spike_trains())
# ISI_e4 = [x for x in ISI_e4 if x != []]
# cvs_e4 = [ np.std(i)/np.mean(i) for i in ISI_e4]
# np.savetxt(a+'/cvs_e4.txt', cvs_e4)
# ISI_pv4=ISI(N[1][1],S_pv4.spike_trains())
# ISI_pv4 = [x for x in ISI_pv4 if x != []]
# cvs_pv4 = [ np.std(i)/np.mean(i) for i in ISI_pv4]
# np.savetxt(a+'/cvs_pv4.txt', cvs_pv4)
# ISI_sst4=ISI(N[1][2],S_sst4.spike_trains())
# ISI_sst4 = [x for x in ISI_sst4 if x != []]
# cvs_sst4 = [ np.std(i)/np.mean(i) for i in ISI_sst4]
# np.savetxt(a+'/cvs_sst4.txt', cvs_sst4)
# ISI_vip4=ISI(N[1][3],S_vip4.spike_trains())
# ISI_vip4 = [x for x in ISI_vip4 if x != []]
# cvs_vip4 = [ np.std(i)/np.mean(i) for i in ISI_vip4]
# np.savetxt(a+'/cvs_vip4.txt', cvs_vip4)
#
# ISI_e5=ISI(N[2][0],S_e5.spike_trains())
# ISI_e5 = [x for x in ISI_e5 if x != []]
# cvs_e5 = [ np.std(i)/np.mean(i) for i in ISI_e5]
# np.savetxt(a+'/cvs_e5.txt', cvs_e5)
# ISI_pv5=ISI(N[2][1],S_pv5.spike_trains())
# ISI_pv5 = [x for x in ISI_pv5 if x != []]
# cvs_pv5 = [ np.std(i)/np.mean(i) for i in ISI_pv5]
# np.savetxt(a+'/cvs_pv5.txt', cvs_pv5)
# ISI_sst5=ISI(N[2][2],S_sst5.spike_trains())
# ISI_sst5 = [x for x in ISI_sst5 if x != []]
# cvs_sst5 = [ np.std(i)/np.mean(i) for i in ISI_sst5]
# np.savetxt(a+'/cvs_sst5.txt', cvs_sst5)
# ISI_vip5=ISI(N[2][3],S_vip5.spike_trains())
# ISI_vip5 = [x for x in ISI_vip5 if x != []]
# cvs_vip5 = [ np.std(i)/np.mean(i) for i in ISI_vip5]
# np.savetxt(a+'/cvs_vip5.txt', cvs_vip5)

#
# ISI_e6=ISI(N[3][0],S_e6.spike_trains())
# ISI_e6 = [x for x in ISI_e6 if x != []]
# cvs_e6 = [ np.std(i)/np.mean(i) for i in ISI_e6]
# np.savetxt(a+'/cvs_e6.txt', cvs_e6)
# ISI_pv6=ISI(N[3][1],S_pv6.spike_trains())
# ISI_pv6 = [x for x in ISI_pv6 if x != []]
# cvs_pv6 = [ np.std(i)/np.mean(i) for i in ISI_pv6]
# np.savetxt(a+'/cvs_pv6.txt', cvs_pv6)
# ISI_sst6=ISI(N[3][2],S_sst6.spike_trains())
# ISI_sst6 = [x for x in ISI_sst6 if x != []]
# cvs_sst6 = [ np.std(i)/np.mean(i) for i in ISI_sst6]
# np.savetxt(a+'/cvs_sst6.txt', cvs_sst6)
# ISI_vip6=ISI(N[3][3],S_vip6.spike_trains())
# ISI_vip6 = [x for x in ISI_vip6 if x != []]
# cvs_vip6 = [ np.std(i)/np.mean(i) for i in ISI_vip6]
# np.savetxt(a+'/cvs_vip6.txt', cvs_vip6)
#



#THIS IS FOR THE VISIMPLE VISUALISATION
# In[39]:
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

# In[40]:

#CHECKING IF I WROTE EVERYTHING CORRECTLY
# import numpy as np
# my_csv = np.genfromtxt('activity_vis.csv', delimiter=',')
# index = my_csv[:, 0]
# spike_time = my_csv[:, 1]
# print(index)
# print(spike_time)

# In[41]:

# print(len(R_vip1.rate))
# print(len(R_e23.t))
# print(R_vip1.rate/Hz)

# In[42]:

#SAVE THE RATES (no need, I can plot the rate myself using only the spike data)
#Write to files the rates.
#In each file I have the rate of the population at each time step.

#layer1
# f=open(b+"/R_vip1rate.txt",'w+') #create the file
# for i in range(0,len(R_vip1.rate)):
#     f.write('%f ' %R_vip1.rate[i])
#     f.write('\n')
# f.close()
#
# #layer 2/3
# f=open(b+"/R_e23rate.txt",'w+') #create the file
# for i in range(0,len(R_e23.rate)):
#     f.write('%f ' %R_e23.rate[i])
#     f.write('\n')
# f.close()
#
# f=open(b+"/R_pv23rate.txt",'w+') #create the file
# for i in range(0,len(R_pv23.rate)):
#     f.write('%f ' %R_pv23.rate[i])
#     f.write('\n')
# f.close()
#
# f=open(b+"/R_sst23rate.txt",'w+') #create the file
# for i in range(0,len(R_sst23.rate)):
#     f.write('%f ' %R_sst23.rate[i])
#     f.write('\n')
# f.close()
#
# f=open(b+"/R_vip23rate.txt",'w+') #create the file
# for i in range(0,len(R_vip23.rate)):
#     f.write('%f ' %R_vip23.rate[i])
#     f.write('\n')
# f.close()
#
# #layer4
# f=open(b+"/R_e4rate.txt",'w+') #create the file
# for i in range(0,len(R_e4.rate)):
#     f.write('%f ' %R_e4.rate[i])
#     f.write('\n')
# f.close()
#
# f=open(b+"/R_pv4rate.txt",'w+') #create the file
# for i in range(0,len(R_pv4.rate)):
#     f.write('%f ' %R_pv4.rate[i])
#     f.write('\n')
# f.close()
#
# f=open(b+"/R_sst4rate.txt",'w+') #create the file
# for i in range(0,len(R_sst4.rate)):
#     f.write('%f ' %R_sst4.rate[i])
#     f.write('\n')
# f.close()
#
# f=open(b+"/R_vip4rate.txt",'w+') #create the file
# for i in range(0,len(R_vip4.rate)):
#     f.write('%f ' %R_vip4.rate[i])
#     f.write('\n')
# f.close()
#
# #layer 5
# f=open(b+"/R_e5rate.txt",'w+') #create the file
# for i in range(0,len(R_e5.rate)):
#     f.write('%f ' %R_e5.rate[i])
#     f.write('\n')
# f.close()
#
# f=open(b+"/R_pv5rate.txt",'w+') #create the file
# for i in range(0,len(R_pv5.rate)):
#     f.write('%f ' %R_pv5.rate[i])
#     f.write('\n')
# f.close()
#
# f=open(b+"/R_sst5rate.txt",'w+') #create the file
# for i in range(0,len(R_sst5.rate)):
#     f.write('%f ' %R_sst5.rate[i])
#     f.write('\n')
# f.close()
#
# f=open(b+"/R_vip5rate.txt",'w+') #create the file
# for i in range(0,len(R_vip5.rate)):
#     f.write('%f ' %R_vip5.rate[i])
#     f.write('\n')
# f.close()
#
# #layer6
# f=open(b+"/R_e6rate.txt",'w+') #create the file
# for i in range(0,len(R_e6.rate)):
#     f.write('%f ' %R_e6.rate[i])
#     f.write('\n')
# f.close()
#
# f=open(b+"/R_pv6rate.txt",'w+') #create the file
# for i in range(0,len(R_pv6.rate)):
#     f.write('%f ' %R_pv6.rate[i])
#     f.write('\n')
# f.close()
#
# f=open(b+"/R_sst6rate.txt",'w+') #create the file
# for i in range(0,len(R_sst6.rate)):
#     f.write('%f ' %R_sst6.rate[i])
#     f.write('\n')
# f.close()
#
# f=open(b+"/R_vip6rate.txt",'w+') #create the file
# for i in range(0,len(R_vip6.rate)):
#     f.write('%f ' %R_vip6.rate[i])
#     f.write('\n')
# f.close()
#


#------------------------------------------------------------------------------
# Show results population rates for all groups
#------------------------------------------------------------------------------

# In[43]:

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


# In[44]:
#TO PLOT IT DIRECTLY
wid=50
rate_plots_noI(R_vip1,R_e23,R_pv23,R_sst23,R_vip23,
             R_e4,R_pv4,R_sst4,R_vip4,
             R_e5,R_pv5,R_sst5,R_vip5,
             R_e6,R_pv6,R_sst6,R_vip6,wid)
