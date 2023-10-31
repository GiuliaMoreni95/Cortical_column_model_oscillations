# File with the Neuronal and network parameters
# Coding style similar to NEST implementation of Potjans

from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
import os
from pprint import pprint
from network_parameters import net_dict, neuron_dict, eqs_dict, receptors_dict

print('---------------------------------')
print('      CORTICAL COLUMN MODEL')
print('---------------------------------')


np.random.seed(net_dict['Seed']) # Set the seed of the simulation for reproducibility

# Need to define the terms in the equations here, because we only import the equations, with no
# values for the parameters inside, so the program won't be able to understand them from network_parameters.py
g_ampa_ext = receptors_dict['g_ampa_ext']
V_E = receptors_dict['V_E']
g_ampa_rec = receptors_dict['g_ampa_rec']
g_gaba = receptors_dict['g_gaba']
g_nmda = receptors_dict['g_nmda']
Mg2 = receptors_dict['Mg2']
alpha = receptors_dict['alpha']
tau_ampa = receptors_dict['tau_ampa']
tau_gaba = receptors_dict['tau_gaba']
tau_nmda_decay = receptors_dict['tau_nmda_decay']
tau_nmda_rise = receptors_dict['tau_nmda_rise']


# Need to create neuronal popualtions outside the class because Brian2 doesn't work well with doing that inside classes
print('---------------------------------')
print('Creating neuronal populations')
print('---------------------------------')
pops = []
num_pops = len(net_dict['populations'])
num_neurons = net_dict['num_neurons']
for m in np.arange(num_pops):
    Vth = neuron_dict['V_th'][m] # neuron_dict[V_th] will be a list or array of thresholds to hold all populations
    Vrest = neuron_dict['V_reset'][m] # neuron_dict[V_reset] will be a list or array of reset to hold all populations
    V_L_i = neuron_dict['V_L'][m] # Same as above
    C_m_i = neuron_dict['C_m'][m] # Same as above
    g_L_i = neuron_dict['g_L'][m] # Same as above
            #tau_ref = self.neuron_dict['tau_ref'][i] # neuron_dict[tau_ref] will be a list or array of refractory to hold all populations
    population = NeuronGroup(num_neurons[m], model=eqs_dict['eqs_neuron'], threshold='v > v_th', # repr method to convert a Quantity object, which is a physical quantity with units, to a string that can be used in a string format expression.
                            reset='v = v_rest', refractory=neuron_dict['tau_ref'][m], method='euler')
            #population.v[0] = self.neuron_dict['V_0'][i] # will be a list or array of initial voltages to hold all populations
    for n in range(num_neurons[m]):
        population[n].v_th = neuron_dict['V_th'][m]
        population[n].v_rest = neuron_dict['V_reset'][m]
        population[n].v[0] = neuron_dict['V_0'][m]
    
    population.V_L = V_L_i # Set V_l for each population as in network_parameters
    population.C_m = C_m_i # Set C_m for each population as in network_parameters
    population.g_L = g_L_i # Set g_L for each population as in network_parameters
    population.V_I = receptors_dict['V_I'][m] # V_I for GABA
    population.I_DC_input = 0*pA # Initial DC input at time t = 0 set to 0pA, can be changed later in simulate function
    pops.append(population)

class Network_main:
    """"
    Parameters
    ---------
    net_dict : dictionary
         Contains parameters specific to overall network (see: 'network_parameters.py').
    neuron_dict : dictionary
         Contains parameters specific to neuron used (see: 'network_parameters.py').
    receptors_dict : dictionary
         Contains parameters specific to AMPA, GABA, NMDA receptors (see: 'network_parameters.py').
    eqs_dict : dictionary
         Contains equations of neurons and specific to AMPA, GABA, NMDA receptors (see: 'network_parameters.py').
    """
    def __init__(self, net_dict, neuron_dict, receptors_dict, eqs_dict):
        self.net_dict = net_dict
        self.neuron_dict = neuron_dict
        self.receptors_dict = receptors_dict
        self.eqs_dict = eqs_dict
        

        self.num_pops = len(self.net_dict['populations']) # Number of populations
        print('Number of populations:', self.num_pops)
        self.num_neurons = self.net_dict['num_neurons'] # Number of neurons
        print(self.net_dict['num_neurons'])

    def create(self):
        """ Creates network populations and input.

        Neuronal populations and devices (recording and generators) are created.

        """
        # self.__create_neuronal_populations()
        self.__create_poisson_bg_input()

    def connect(self):
        """ Connects the population and devices (recording and generators).

        Neuronal populations are connected between and within themselves,
        as well as generator devices and recording devices

        """
        
        self.__connect_poisson_bg_input()
        self.__connect_neuronal_populations()
        self.__connect_recording_devices()

    def simulate(self, t_sim):
        """ Simulates the network and plots results

        Parameters
        ----------
        t_sim: number
            Simulation time in ms.

        """
        @network_operation(dt=1*ms)
        def update_input(t):
            if t>self.net_dict['t_start_DC'] and t<self.net_dict['t_end_DC']: # Add DC input to E4 starting at time t_start_DC and end at t_end_DC
                pops[5].I_DC_input = self.net_dict['I_DC'][5] # Value corresponding to E4
                #pops[insert_desired_population_index].I_DC_input = self.net_dict['I_DC'][insert_desired_population_index] # Value corresponding to insert_desired_population_index

            else:
                pops[5].I_DC_input = 0.*pA
                #pops[insert_desired_population_index].I_DC_input = 0.*pA

        print('-----------------------------')
        print('Simulating')
        print('-----------------------------')
        # Here pay attention to add all synapses and objects (incl network operations) to self.net_mon, otherwise they won't be included in the sim!
        self.net_run = Network(update_input, self.spike_mon[:], self.rate_mon[:], pops, self.Poisson_groups[:], self.S_Poisson[:],
                                self.S_ampa[:], self.S_gaba[:], self.S_nmda[:]) # This uses Brian2 Network class to inlcude the monitors when running the simulation
        self.net_run.run(t_sim)

        print('Finished simulation')
        print('-----------------------------')


    def __create_neuronal_populations(self):
        """ Creates the neuronal populations with parameters defined in 'network_parameters.py'.

            Stores them in a list, pops
        """
        #print('Creating neuronal populations')

        #pops = pops # Class variable to hold populations and work inside the class with

    def __create_poisson_bg_input(self):
        """ Creates Poisson background input ith parameters 
        as specified in 'network_parameters.py'.

            The same number of generators is created as neuronal populations.
            The number of neurons in each Poisson generator is equal to the
            number of neurons in the population it targets.

        """
        print('-----------------------------')
        print('Creating Poisson background input')
        print('-----------------------------')

        self.Poisson_groups = [] # List to hold Poisson background generators
        for n in np.arange(self.num_pops):
            Poisson_groups = PoissonGroup(self.net_dict['num_neurons'][n], rates=self.net_dict['bg_rate'][n])
            self.Poisson_groups.append(Poisson_groups)
    
    def __connect_neuronal_populations(self):
        """ Connects neuronal populations recurrently, with parameters
        as specified in 'network_parameters.py'.

            There are E-E, E-I, I-E and I-I connections.

        """
        print('-----------------------------')
        print('Connecting synapses (289)')
        print('-----------------------------')

        self.S_ampa = []
        self.S_nmda = []
        self.S_gaba = []
        # Excitatory are: [1, 5, 9, 13]
        # Inhibitory are: [0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16]
        
        # Dictionary storing every population's equation and what should happen with every pre-synaptic spike
        # Excitatory populations have both AMPA and NMDA parameters
        pop_map = {
            0: ('INH', 'eqs_s_gaba_l1vip', ''),
            1: ('EXC', 'eqs_s_ampa_l23', 'eqs_s_nmda_l23'),
            2: ('INH', 'eqs_s_gaba_l23pv', ''),
            3: ('INH', 'eqs_s_gaba_l23sst', ''),
            4: ('INH', 'eqs_s_gaba_l23vip', ''),
            5: ('EXC', 'eqs_s_ampa_l4', 'eqs_s_nmda_l4'),
            6: ('INH', 'eqs_s_gaba_l4pv', ''),
            7: ('INH', 'eqs_s_gaba_l4sst', ''),
            8: ('INH', 'eqs_s_gaba_l4vip', ''),
            9: ('EXC', 'eqs_s_ampa_l5', 'eqs_s_nmda_l5'),
            10: ('INH', 'eqs_s_gaba_l5pv', ''),
            11: ('INH', 'eqs_s_gaba_l5sst', ''),
            12: ('INH', 'eqs_s_gaba_l5vip', ''),
            13: ('EXC', 'eqs_s_ampa_l6', 'eqs_s_nmda_l6'),
            14: ('INH', 'eqs_s_gaba_l6pv', ''),
            15: ('INH', 'eqs_s_gaba_l6sst', ''),
            16: ('INH', 'eqs_s_gaba_l6vip', ''),
        }
        iteration = 1
        # For each target, create synapses from all possible sources
        for n, target_pop in enumerate(pops):
            for m, source_pop in enumerate(pops, start=0):
                print('Iteration', iteration)
                iteration += 1
                neuron_type, eq_1, eq_2 = pop_map[m]
                # GABA synapses
                if neuron_type == 'INH':
                    S = Synapses(source_pop, target_pop, model=self.eqs_dict[eq_1], on_pre='s_gaba += 1', method='euler')
                    if m == n:
                        S.connect(condition='i != j', p=self.net_dict['connect_probs'][n][m]) # Prevent auto-synapses
                    else:
                        S.connect(p=self.net_dict['connect_probs'][n][m])
                    if self.net_dict['connect_probs'][n][m] == 0 or self.net_dict['synaptic_strength'][n][m] == 0: # Condition to prevent division by 0 when calculating weight
                        S.w = 0
                    else:
                        S.w = self.net_dict['global_g']*self.net_dict['synaptic_strength'][n][m]/(self.net_dict['num_neurons'][m]*self.net_dict['connect_probs'][n][m]) # Formula for synaptic weight used in Giulia's model
                    S.delay = self.net_dict['delay']
                    self.S_gaba.append(S)
                    del S
                elif neuron_type == 'EXC':
                    # AMPA synapses
                    S = Synapses(source_pop, target_pop, model=self.eqs_dict[eq_1], on_pre='s_ampa += 1', method='euler')
                    if m == n:
                        S.connect(condition='i != j', p=self.net_dict['proportion_AMPA']*self.net_dict['connect_probs'][n][m]) # Prevent auto-synapses
                    else:
                        S.connect(p=self.net_dict['proportion_AMPA']*self.net_dict['connect_probs'][n][m])
                    if self.net_dict['connect_probs'][n][m] == 0 or self.net_dict['synaptic_strength'][n][m] == 0: # Condition to prevent division by 0 when calculating weight
                        S.w = 0
                    else:
                        S.w = self.net_dict['global_g']*self.net_dict['synaptic_strength'][n][m]/(self.net_dict['num_neurons'][m]*self.net_dict['proportion_AMPA']*self.net_dict['connect_probs'][n][m]) # Formula for synaptic weight used in Giulia's model
                    S.delay = self.net_dict['delay']
                    self.S_ampa.append(S)
                    del S

                    # NMDA synapses
                    S = Synapses(source_pop, target_pop, model=self.eqs_dict[eq_2], on_pre='x += 1', method='euler')
                    if m == n:
                        S.connect(condition='i != j', p=self.net_dict['proportion_NMDA']*self.net_dict['connect_probs'][n][m]) # Prevent auto-synapses
                    else:
                        S.connect(p=self.net_dict['proportion_NMDA']*self.net_dict['connect_probs'][n][m])
                    if self.net_dict['connect_probs'][n][m] == 0 or self.net_dict['synaptic_strength'][n][m] == 0: # Condition to prevent division by 0 when calculating weight
                        S.w = 0
                    else:
                        S.w = self.net_dict['global_g']*self.net_dict['synaptic_strength'][n][m]/(self.net_dict['num_neurons'][m]*self.net_dict['proportion_NMDA']*self.net_dict['connect_probs'][n][m]) # Formula for synaptic weight used in Giulia's model
                    S.delay = self.net_dict['delay']
                    self.S_nmda.append(S)
                    del S


    def __connect_poisson_bg_input(self):
        """ Connects generator devices as specified in 'network_parameters.py'.

            Connects Poisson background generator to each population.
        
        """
        print('Connecting Poisson background input (16) ')
        print('-----------------------------')

        self.S_Poisson = [] # List for Poisson background synapses
        for n in np.arange(self.num_pops):
            print('Connecting Poisson', n)
            S_Poisson = Synapses(self.Poisson_groups[n], pops[n], on_pre='s_ampa_ext += 1', method='euler')
            self.S_Poisson.append(S_Poisson)
            self.S_Poisson[n].connect(j='i') # Connects Poisson one-to-one

    def __connect_recording_devices(self):
        """ Creates the spike/voltage recording devices for each population.

            It automatically connects to the population specified.

        """
        print('-----------------------------')
        print('Connecting recording devices')

        self.spike_mon = list(range(self.num_pops)) # List of spike monitors
        self.rate_mon = list(range(self.num_pops)) # List of rate monitors
        for i in np.arange(self.num_pops):
            self.spike_mon[i] = SpikeMonitor(pops[i]) # Record from all neurons in each population
            self.rate_mon[i] = PopulationRateMonitor(pops[i])
        
    # def poisson_external_input(self):
    #     """ Creates Poisson input for external stimulation.
    #         Connects relevant inputs to targeted neuronal populations.

    #     """
    #     self.Poisson_ex = [] # List to hold Poisson input generators
    #     # Create potential Poisson generators for each neuronal population, but only connect those relevant
    #     for n in np.arange(self.num_pops):
    #         Poisson_ex_n = PoissonGroup(self.net_dict['num_neurons'][n], rates=self.net_dict['Poisson_external'][n])
    #         self.Poisson_ex.append(Poisson_ex_n)
        
    #     # Connect relevant populations
    #     # But make it only at a specific time
    #     self.S_Poisson_ex = [] # List for Poisson background synapses
    #     poisson_targets = [5]
    #     for n in poisson_targets:
    #         print('Connecting Poisson external input', n)
    #         S_Poisson_input = Synapses(self.Poisson_ex[n], pops[n], on_pre='s_ampa_ext += 1', method='euler')
    #         self.S_Poisson_ex.append(S_Poisson_input)
    #         self.S_Poisson_ex[n].connect(j='i') # Connects Poisson one-to-one
    
    def plot_spikes(self):
        """ Plots raster graphs after simulation is run.

        """
        current_directory = os.getcwd()
        folder_name = "Data"
        folder_path = os.path.join(current_directory, folder_name)
        list_ex = [1, 5, 9 , 13] # Indexes corresponding to excitatory populations
        color = ''
        for i in range(len(pops)):
            pop = self.net_dict['populations'][i]
            if i in list_ex:
                color = '.r' # Red for excitatory
            else:
                color = '.b' # Blue for inhibitory
            plt.figure(i)
            plt.plot(self.spike_mon[i].t/ms, self.spike_mon[i].i, color, label=pop)
            xlabel('Time (ms)')
            ylabel('Neuron index')
            legend()
            plt.savefig(os.path.join(folder_path, f"5000_spikes_population_{i}.png"))

    def firing_rates(self, t_sim):
        """ Calculates and prints mean firing rates in spikes/s for each population.

        """
        for i in range(self.num_pops):
            mean_rate = self.spike_mon[i].num_spikes/(num_neurons[i]*t_sim)
            print(f"Mean rate of population {self.net_dict['populations'][i]}: {mean_rate:.2f} Hz")

    # Function not yet finished
    def firing_rates_evoked(self, t_sim):
        """ Calculates and prints mean firing rates in spikes/s for each population after stimulus onset.

        """
        for i in range(self.num_pops):
            mean_rate = self.spike_mon[i].num_spikes/(num_neurons[i]*t_sim)
            print(f"Mean rate of population {self.net_dict['populations'][i]}: {mean_rate:.2f} Hz")
            

    def write_data(self):
        """ Creates files with spike data.

        """

        a = 'Sp_base_output'

        for n in range(self.num_pops):
            # Write total number of spikes
            f=open(a + f"/{self.net_dict['populations'][n]}numspikes.txt",'w+') # Create the file
            f.write('%f ' %self.spike_mon[n].num_spikes)
            f.close()

            # Save indexes of spikes in 'i' files
            f=open(a + f"/{self.net_dict['populations'][n]}i.txt",'w+') # Create the file
            for i in range(0, len(self.spike_mon[n].i)):
                f.write('%i ' %self.spike_mon[n].i[i])
                f.write('\n')
            f.close()

            # Save corresponding spike time in 't' files
            f=open(a + f"/{self.net_dict['populations'][n]}t.txt",'w+') # Create the file
            for i in range(0,len(self.spike_mon[n].t)):
                f.write('%f ' %self.spike_mon[n].t[i])
                f.write('\n')
            f.close()
