# File with the Neuronal and network parameters
# V0 should be drawn from standard distributions, same as weights and delays

from brian2 import *
import network_main
from network_parameters import net_dict, neuron_dict, receptors_dict, eqs_dict

defaultclock.dt = 0.1*ms # Time resolution of the simulation
t_sim = 500*ms # Simulation time

net = network_main.Network_main(net_dict, neuron_dict, receptors_dict, eqs_dict)

net.create()
net.connect()
# net.poisson_external_input()
net.simulate(t_sim)

#net.plot_spikes() #To save the plots you need to create a folder in the directory called 'Sp_base_output'
net.firing_rates(t_sim)
#net.write_data() #To save the data plots you need to create a folder in the directory called 'Data'
