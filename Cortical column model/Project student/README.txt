Welcome! 

Here a new version of the cortical column code is provided. 

This code reproduce the results of the cortical column with fixed weights (no STDP rule implemented). 
This code (created by Rares Dorcioman) is a different version of 'MAIN_CODE1_fixed_weights.py'.
The results are identical. 
The 'import_files' are not needed, all the parameters of the neurons are already inside the code.
With this new version there is no need to download any additional file. 
 
There are three .py codes:
- network_main.py
- network_parameters.py
- simulate.py
To run the model you only need to have this 3 files in a folder and just run "simulate.py" in your terminal (just type: 'python simualate.py').
You don't need any additional data. 


If you want to change the input to a particular neuron group:
In 'network_parameters.py' in the dictionary 'net_dict' there is a parameter called 'I_DC'. 
It is is indexed as VIP1, E2/3, PV2/3, SST2/3, VIP2/3, E4, PV4 etc. (the order is: E,PV,SST,VIP in layer 2/3,4,5,6) 
So the index for E4 is '5' (first index is 0).
Now the input is set to 0, but you can change that number to a negative value to send an excitatatory current to E4. 
You can also change the values there to send desired inputs to other specific populations. 
The input will start at t_start_DC and end at t_end_DC. These times can also be specified there (just below 'I_DC', see code).

If you want to send inputs to more populations (not just E4) you should also change the following in 'network_main.py'.
In the function 'simulate', below @network_operation you should uncomment 'pop[insert_desired_population_index]' and insert
the index of the population you want to send the input to.

Best,
Giulia Morneni
