# File with the simulations. 
# It imports the network parameters and the network structure

from brian2 import *
import network_main
from network_parameters import net_dict, neuron_dict, receptors_dict, eqs_dict

defaultclock.dt = 0.1*ms # Time resolution of the simulation
t_sim = 2*ms # Simulation time

net = network_main.Network_main(net_dict, neuron_dict, receptors_dict, eqs_dict)

net.create()
net.connect()
# net.poisson_external_input()
net.simulate(t_sim)

#net.plot_spikes()
#net.firing_rates(t_sim)
#net.write_data()
