#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#With this program I distribute the neurons in a 3D space. I then can choose where to put my electrode. 
#I count how many neurons I have in the sphere of resolution of the electrode and save it.
# I also assign a weight to each neuron depending on the distance from the electrode (1/r law)
#This program produce some files which are then necessary for the main program.
#The weghts files and number of neurons and position of the electrode


# In[1]:


import numpy as np
import time
import matplotlib.pyplot as plt


# In[2]:


Ntot=5000
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
print(np.sum(N))
print("+ in layer 1 we have:%s neurons"%N1)
print('--------------------------------------------------')


# In[3]:


# I am interested only in excitatory neurons, I save how many for each layer
N2_3e=N[0][0]
N4e=N[1][0]
N5e=N[2][0]
N6e=N[3][0]

#Layers of my column
l23_min=0   #μm 
l23_max=250 #μm 
l4_max=450  #μm 
l5_max=775 #μm 
l6_max=1150 #μm 

#Diameter of cortical column: 300 μm to 600 μm
#I choose diameter of 400 μm so radius=200 μm
rad_cortical=200 # μm
rad=rad_cortical/10 # I need this to then multiply r and have a point in the right range

#I generate the x coordinates for the 4 layers
A=np.random.uniform(l23_min, l23_max, N2_3e)
B=np.random.uniform(l23_max, l4_max, N4e )
C=np.random.uniform(l4_max, l5_max, N5e )
D=np.random.uniform(l5_max, l6_max, N6e )

#arrays containing the positions of all neurons initialized at 0.
xdata=[]
ydata=[]
zdata=[]

#f = open("network_new1.csv", "w")
#I distribute the excitatory neurons of the 4 layers in 3d
for i in range(0,N2_3e):
    r=np.random.uniform(0, 100) 
    theta=np.random.uniform(0, 2*np.pi) #distribute in the circonference at that high
    R=np.sqrt(r)*rad #I use the sqrt of r so that I have uniform distribution in the cylinder.
    #If I don't use this, I am distributing neurons in the cylinder not in uniform way: 
    #there will be same number outside and inside so more dense inside and less putside (because outside is bigger)
    xdata.append(A[i])
    ydata.append(R*np.cos(theta))
    zdata.append(R*np.sin(theta))
    #f.write('%i,%f,%f,%f\n'%(i,A[i],R*np.cos(theta),R*np.sin(theta)))
for i in range(0,N4e):
    r=np.random.uniform(0, 100)
    theta=np.random.uniform(0, 2*np.pi)
    R=np.sqrt(r)*rad
    xdata.append(B[i])
    ydata.append(R*np.cos(theta))
    zdata.append(R*np.sin(theta))
    #f.write('%i,%f,%f,%f\n'%(N2_3+i,B[i],R*np.cos(theta),R*np.sin(theta)))
for i in range(0,N5e):
    r=np.random.uniform(0, 100)
    theta=np.random.uniform(0, 2*np.pi)
    R=np.sqrt(r)*rad
    xdata.append(C[i])
    ydata.append(R*np.cos(theta))
    zdata.append(R*np.sin(theta))
    #f.write('%i,%f,%f,%f\n'%(N2_3+N4+i,C[i],R*np.cos(theta),R*np.sin(theta)))
for i in range(0,N6e):
    r=np.random.uniform(0, 100)
    theta=np.random.uniform(0, 2*np.pi)
    R=np.sqrt(r)*rad
    xdata.append(D[i])
    ydata.append(R*np.cos(theta))
    zdata.append(R*np.sin(theta))
    #f.write('%i,%f,%f,%f\n'%(N2_3+N4+N5+i,D[i],R*np.cos(theta),R*np.sin(theta)))

#f.close()


# In[4]:


#Visualisation of the excitatory neurons
fig = plt.figure(figsize=(15,10))
ax = plt.axes(projection='3d')
ax.scatter3D(xdata, ydata, zdata, c=xdata, cmap='viridis', linewidth=0.5)
ax.set_xlabel('X', fontsize=20)
ax.set_ylabel('Y',fontsize=20)
ax.set_zlabel('Z', fontsize=20)
#ax.view_init(100, 30)


# In[5]:


#Function to count the neurons in each subsection of the sphere and compute their weight depending on the distance from the electrode
def weight_LFP(depth,radius,alpha,rad_action,xdata,ydata,zdata):
    
    #position of the electrode
    x_el=depth
    y_el=radius*np.cos(alpha)
    z_el=radius*np.sin(alpha)
    
    #total number of excitatory neurons
    Ne=N2_3e+N4e+N5e+N6e
    
    #When I find a neuron in the sphere I look in which layer it is and count it.
    n_neurons=np.zeros(4) # I have 4 n, one for each layer2. I initialize at 0
    
    #arrays that contain the weight of each neuron inside the resolution sphere of the electrode
    weights23=[]
    weights4=[]
    weights5=[]
    weights6=[]
    #I save the weights in files
    f23 = open("LFP_files/weight23.txt", "w")
    f4 = open("LFP_files/weight4.txt", "w")
    f5 = open("LFP_files/weight5.txt", "w")
    f6 = open("LFP_files/weight6.txt", "w")
    
    for i in range(0,Ne):
        distance=np.sqrt((xdata[i]-x_el)**2+(ydata[i]-y_el)**2+(zdata[i]-z_el)**2) #I compute the distance from the electrode
        if distance < rad_action: #If the neuron is inside the sphere I look exactly where it is (which layer and update the count of that layer)
            #print(xdata[i])
            if xdata[i]<l23_max: 
                n_neurons[0]+= 1 #count +1 if the neuron is in layer 2/3
                weights23.append(1/distance)
                f23.write('%f \n'%(1/distance)) #save the values of the weight
            elif xdata[i] < l4_max:
                n_neurons[1]+= 1
                weights4.append(1/distance)
                f4.write('%f \n'%(1/distance))
            elif xdata[i] < l5_max:
                n_neurons+= 1
                weights5.append(1/distance)
                f5.write('%f \n'%(1/distance))
            elif xdata[i] < l6_max:
                n_neurons[3]+= 1
                weights6.append(1/distance)
                f6.write('%f \n'%(1/distance))
    f23.close()
    f4.close()
    f5.close()
    f6.close()
    print(n_neurons)
#     print(len(weights23))
#     print(len(weights4))
#     print(len(weights5))
#     print(len(weights6)) #ok same len of the numbers in n_neurons. Working

#write the number of neurons in each layer inside the sphere.
    f = open("LFP_files/numberLFP.txt", "w")
    for i in range(4):        
        f.write('%f \n'%(n_neurons[i]))
    f.close()


# In[14]:


# l23_min=0   #μm 
# l23_max=250 #μm 
# l4_max=450  #μm 
# l5_max=775 #μm 
# l6_max=1150 #μm 

rad_action= 100 #area of recording of the electrode (50-350)
depth=200   #should be between 0 and 1150 (l23-l6)
radius=0   #should be between 0 and 200 (center of the column, edge)
alpha=0     #should be between 0 and 2*pi

#I write on a file the chosed electrode data
f = open("LFP_files/electrode_data.txt", "w")
f.write('%f \n'%(rad_action))
f.write('%f \n'%(depth))
f.write('%f \n'%(radius))
f.write('%f \n'%(alpha))
f.close()

#I call the function to compute the weight of LFP based on the choice of el_data
weights_comput=weight_LFP(depth,radius,alpha,rad_action,xdata,ydata,zdata)


# In[7]:


#IN THE MAIN PROGRAM I WILL THEN HAVE:


# In[8]:


# #In the main program I will need to import these data
# el_data=np.array(np.loadtxt('electrode_data.txt') )
# rad_action= el_data[0]
# depth=el_data[1]  
# radius=el_data[2]  
# alpha=el_data[3]    

# print(rad_action)
# print(depth)
# print(radius)
# print(alpha)


# In[9]:


# #number of neurons from which I need to record
# num=np.array(np.loadtxt('numberLFP.txt') )


# In[10]:


# #Import the weights computed 
# W23=np.array(np.loadtxt('weight23.txt') )
# W4=np.array(np.loadtxt('weight4.txt') )
# W5=np.array(np.loadtxt('weight5.txt') )
# W6=np.array(np.loadtxt('weight6.txt') )


# In[11]:


# #Compute LFP cotributions from a subset of neurons
# def LFP_contribution(igabaE,iampaE,inmdaE,iampaextE,weight,Num_neu,steps,tau,alpha):

#     gaba_all=[0 for i in range(0,int(steps-tau))]
#     ampa_all=[0 for i in range(0,int(steps-tau))]
#     nmda_all=[0 for i in range(0,int(steps-tau))]
#     ampaext_all=[0 for i in range(0,int(steps-tau))]
#     if Num_neu !=0 :
#         for i in range(0,Num_neu):
#             gaba_all+=weight[i]*np.array(-igabaE.I_GABA[i][:-tau]/pA)
#             ampa_all+=weight[i]*np.array(-iampaE.I_AMPA_rec[i][tau:]/pA)
#             nmda_all+=weight[i]*np.array(-inmdaE.I_NMDA[i][:-tau]/pA)
#             ampaext_all+=weight[i]*np.array(-iampaextE.I_AMPA_ext[i][:-tau]/pA)

        
#     #LFP_array=ampa_all - alpha*gaba_all #it'a an array
#     LFP_array= ampaext_all + ampa_all - alpha*gaba_all + nmda_all #SUM OF I (ABS)
    
#     return LFP_array

# # I will call this function (above) for each layer where I have > 0 neurons inside the sphere
# #I will sum them togheter and then call the function below to have a normalized LFP


# In[12]:


# def LFP_final(LFP_summed):
   
    
#     import statistics
#     std_LFP=statistics.stdev(LFP_summed)
#     print(std_LFP)
#     LFP_array_norm= (LFP_summed-np.mean(LFP_summed))/std_LFP

# #     #Check that is 1
# #     std_LFPnorm=statistics.stdev(LFP_array_norm)
# #     print(std_LFPnorm)

#     return LFP_array_norm


# In[13]:


# #Another way to read from files
# with open("weight23.txt", "r") as f:
#     weight23 = f.readlines()
# print(len(weight23))
# W23=np.zeros(len(weight23))
# for i in range(len(weight23)):
#     W23[i]=float(weight23[i])
# print(W23) #It is an np.array

# with open("weight4.txt", "r") as f:
#     weight4 = f.readlines()
# print(len(weight4))
# W4=np.zeros(len(weight4))
# for i in range(len(weight4)):
#     W4[i]=float(weight4[i])
# #print(W4) #It is an np.array

# with open("weight5.txt", "r") as f:
#     weight5 = f.readlines()
# print(len(weight5))
# W5=np.zeros(len(weight5))
# for i in range(len(weight5)):
#     W5[i]=float(weight5[i])
# #print(W5) #It is an np.array

# with open("weight6.txt", "r") as f:
#     weight6 = f.readlines()
# print(len(weight6))
# W6=np.zeros(len(weight6))
# for i in range(len(weight6)):
#     W6[i]=float(weight6[i])
# print(W6) #It is an np.array


# with open("numberLFP.txt", "r") as f:
#     numbers = f.readlines()
# #print(numbers) # list of strings

# num=np.zeros(4)
# for i in range(4):
#     num[i]=float(numbers[i]) #I create a list of floats
# print(num)


# In[ ]:




