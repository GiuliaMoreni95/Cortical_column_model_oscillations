#!/usr/bin/env python
# coding: utf-8

# In[1]:


#PROGRAM with all 3 receptors implemented. Different populations are connected (like we had multiple layers)
#We use the exact equations for GABA,AMPA,NMDA (with summed variable) != from the program without summed.
# It was done to check on the SPEED. 32 (non summed) s vs 42(summed)


# In[2]:

from brian2 import *
import numpy as np
import time
import matplotlib.pyplot as plt
from brian2tools import *
from brian2 import profiling_summary
import statistics


startbuild = time.time()

#model var
Vth= -55*mV
Vrest= -70*mV
tau_ref= 2* ms
NE=2
NPV=1

g_AMPA_rec_I = 1.0 * nS
g_AMPA_ext_I = 1.0 *nS
g_GABA_I= 0.973 * nS

g_AMPA_rec_E = 1.0 * nS
g_AMPA_ext_E = 1.0 *nS
g_GABA_E= 0.973 * nS
g_m_E = 25. * nS
C_m_E = 0.5 * nF

g_NMDA_E = 1.0 * nS
g_NMDA_I = 1.0 * nS
tau_NMDA_rise = 2. * ms
tau_NMDA_decay = 100. * ms
alpha_NMDA = 0.5 / ms
Mg2 = 1.

wextPV=2.0
wextE=1.0

tau_AMPA= 2 *ms
tau_GABA= 5 *ms
g_m_I = 20. * nS
C_m_I = 0.2 * nF
V_L = Vrest
V_E= 0. *mV
V_I = -70. * mV


taupre = 20*ms
taupost = 20*ms
wmax = 10
dApre = 0.01
dApost = -dApre*taupre/taupost*1.05


eqsE='''
    dv / dt = (- g_m_E * (v - V_L) - I_syn) / C_m_E : volt (unless refractory)
    I_syn = I_AMPA_rec + I_AMPA_ext + I_GABA + I_NMDA + I_ext: amp

    I_ext: amp

    I_AMPA_ext= g_AMPA_ext_E * (v - V_E) * wextE * s_AMPA_ext : amp
    ds_AMPA_ext / dt = - s_AMPA_ext / tau_AMPA : 1
    # Here I don't need the summed variable because the neuron receive inputs from only one Poisson generator. Each neuron need only one s.


    I_AMPA_rec = g_AMPA_rec_E * (v - V_E) * 1 * s_AMPA_tot : amp
    s_AMPA_tot : 1  #the eqs_ampa solve many s and sum them and give the summed value here
#Each neuron receives inputs from many neurons. Each of them has his own differential equation s_AMPA (where I have the deltas with the spikes).
#I then sum all the solutions s of the differential equations and I obtain s_AMPA_tot_post.

     I_GABA= g_GABA_E * (v - V_I) * s_GABA_tot : amp
     s_GABA_tot :1


     I_NMDA  = g_NMDA_E * (v - V_E) / (1 + Mg2 * exp(-0.062 * v / mV) / 3.57) * s_NMDA_tot : amp
    s_NMDA_tot : 1


 '''

eqsPV='''

    dv / dt = (- g_m_I * (v - V_L) - I_syn) / C_m_I : volt (unless refractory)
    I_syn = I_AMPA_rec + I_AMPA_ext + I_GABA + I_NMDA + I_ext: amp

    I_ext: amp

    I_AMPA_ext= g_AMPA_ext_I * (v - V_E) * wextPV * s_AMPA_ext : amp
    ds_AMPA_ext / dt = - s_AMPA_ext / tau_AMPA : 1
    # Here I don't need the summed variable because the neuron receive inputs from only one Poisson generator. Each neuron need only one s.


    I_AMPA_rec = g_AMPA_rec_I * (v - V_E) * 1 * s_AMPA_tot : amp
    s_AMPA_tot : 1  #the eqs_ampa solve many s and sum them and give the summed value here
#Each neuron receives inputs from many neurons. Each of them has his own differential equation s_AMPA (where I have the deltas with the spikes).
#I then sum all the solutions s of the differential equations and I obtain s_AMPA_tot_post.

     I_GABA= g_GABA_I * (v - V_I) * s_GABA_tot : amp
     s_GABA_tot :1


     I_NMDA  = g_NMDA_I * (v - V_E) / (1 + Mg2 * exp(-0.062 * v / mV) / 3.57) * s_NMDA_tot : amp
    s_NMDA_tot : 1


 '''

eqs_ampa='''
          s_AMPA_tot_post= w_AMPA* s_AMPA : 1 (summed)
          ds_AMPA / dt = - s_AMPA/ tau_AMPA : 1  (clock-driven)
          w_AMPA: 1
          w_syn: 1
          dapre/dt = -apre/taupre : 1 (clock-driven) #(event-driven)
          dapost/dt = -apost/taupost : 1 (clock-driven) #(event-driven)
        '''

eqs_gaba='''
        s_GABA_tot_post= w_GABA* s_GABA : 1 (summed)
        ds_GABA/ dt = - s_GABA/ tau_GABA : 1 (clock-driven)
        w_GABA: 1
'''

eqs_nmda='''s_NMDA_tot_post = w_NMDA * s_NMDA : 1 (summed)
            ds_NMDA / dt = - s_NMDA / tau_NMDA_decay + alpha_NMDA * x * (1 - s_NMDA) : 1 (clock-driven)
            dx / dt = - x / tau_NMDA_rise : 1 (clock-driven)
            w_NMDA : 1
'''


popE = NeuronGroup(NE, model=eqsE, threshold='v > Vth', reset='v = Vrest', refractory=tau_ref, method='euler')

print(popE[0])
print(popE[1])




popPV= NeuronGroup(NPV, model=eqsPV, threshold='v > Vth', reset='v = Vrest', refractory=tau_ref, method='euler')

for k in range(0,NE):
    popE[k].v[0]=Vrest

for k in range(0,NPV):
    popPV[k].v[0]=Vrest


popE[0].I_ext=-1000*pA
popE[1].I_ext=-10000*pA
popPV.I_ext=-500*pA

#print(popE)
conn= Synapses(popE,popPV,model=eqs_ampa,
               on_pre='''
               s_AMPA+=1
               apre += dApre
               w_AMPA=w_AMPA+apost #w = clip(w + apost, 0, wmax)
               ''',
               on_post='''
               apost += dApost
               w_AMPA=w_AMPA + apre #w = clip(w + apre, 0, wmax)
               ''', method='euler')

conn.connect(i=[0, 1], j=[0])
conn.w_AMPA=1.0
#conn.w_AMPA[0]= 1.0 # I can choose the weight of each synapse.
#conn.w_AMPA[1]= 2.0

#If I switch off the connections I only expect popPV to be governed by noise (POisson group connected)
#If I switch on the connections I expect popPV to be governed by noise (POisson group connected)
#and to be influenced by popE when spike in popE occur
M = StateMonitor(conn, ['w_AMPA', 'apre', 'apost'], record=True)



# conn1= Synapses(popE,popPV,model=eqs_nmda,on_pre='x+=1', method='euler')
# conn1.connect()
# conn1.w_NMDA= 1.0


# conn2= Synapses(popE,popE,model=eqs_ampa,on_pre='s_AMPA+=1', method='euler')
# conn2.connect('i != j')
# conn2.w_AMPA= 1.0


# conn3= Synapses(popE,popE,model=eqs_nmda,on_pre='x+=1', method='euler')
# conn3.connect('i != j')
# conn3.w_NMDA= 1.0


# conn4= Synapses(popPV,popE,model=eqs_gaba,on_pre='s_GABA+=1', method='euler')
# conn4.connect()
# conn4.w_GABA= 1.0

# conn5= Synapses(popPV,popPV,model=eqs_gaba,on_pre='s_GABA+=1', method='euler')
# conn5.connect('i != j')
# conn5.w_GABA= 10.0



rate=4000 * Hz

# ext_inputE= PoissonGroup(1,rates= rate)
# ext_connE = Synapses(ext_inputE, popE[0], on_pre='s_AMPA_ext += 1')
# ext_connE.connect(j='i')

# ext_inputE= PoissonGroup(1,rates= 3000*Hz)
# ext_connE = Synapses(ext_inputE, popE[1], on_pre='s_AMPA_ext += 1')
# ext_connE.connect(j='i')


# ext_inputPV= PoissonGroup(NPV,rates= rate)
# ext_connPV = Synapses(ext_inputPV,popPV,on_pre='s_AMPA_ext += 1')
# ext_connPV.connect(j='i')


mE=StateMonitor(popE, 'v',record=True)
mPV=StateMonitor(popPV, 'v',record=True)

n_activity=10
S_e = SpikeMonitor(popE[:], record=True) #LAYER 2_3
S_pv = SpikeMonitor(popPV[:], record=True)

start = time.time()
run(300*ms)
end = time.time()


# In[3]:



figure(figsize=(4, 8))
subplot(211)
plot(M.t/ms, M.apre[0], label='apre')
plot(M.t/ms, M.apost[0], label='apost')
plot(M.t/ms, M.apre[1], label='apre')
plot(M.t/ms, M.apost[1], label='apost')

#plot(M.t/ms, M.apre[1], label='apre1')
#plot(M.t/ms, M.apost[1], label='apost1')

legend()
subplot(212)
plot(M.t/ms, M.w_AMPA[0], label='w_AMPA')
plot(M.t/ms, M.w_AMPA[1], label='w_AMPA')

legend(loc='best')
xlabel('Time (ms)');




fig2 = plt.figure(figsize=(15,7))
plot(mE.t/ms,mE.v[0],label='e0')
plot(mE.t/ms,mE.v[1],label='e1')
plot(mPV.t/ms,mPV.v[0],label='pv')
xlabel('time (ms)')
ylabel('Membran potential V (mV)')
xlim(0,50)
legend()
show()


f= plt.figure(figsize=(16,9))
# plot(S_e.t/ms,S_e.i+ 1 * n_activity,'.', markersize=2,color='b', label='e')
# plot(S_pv.t/ms,S_pv.i+ 0* n_activity,'.', markersize=2,color='r', label='pv')
plot(S_e.t/ms,S_e.i,'.', markersize=2,color='b', label='e')
plot(S_pv.t/ms,S_pv.i,'.', markersize=2,color='r', label='pv')
xlabel('time (ms)')
legend()
plt.title(('Pops activity - layer 2/3 ({} neurons/pop)'.format(n_activity)))


#print(S_e.t)
#print(S_e.i)
print("tempo:",end-start)


# In[4]:


#print(S_e.spike_trains())


# In[5]:


print(S_e.all_values())


# In[6]:



print(len(S_pv.t))
print(len(S_pv.i))


# In[ ]:
