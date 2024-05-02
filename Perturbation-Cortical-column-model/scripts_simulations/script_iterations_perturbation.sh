#These values are the one that will be used by the main program to run the current simulation
#First value of each list will be used for 1st simulation, 2nd from each list for the second simulation, etc. 
param1_values=(i1 i2 i3 i4 i5 i6 i7 i8 i9 i10 i11 i12 i13 i14 i15 i16) #folder where I save the different simulaitons 
param2_values=(0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3) #layer
param3_values=(0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3) #cell_type

#I want to use the values from the "param_values"
for index in "${!param1_values[@]}"; do
    param1="${param1_values[$index]}" #folder index
    param2="${param2_values[$index]}" #layer index
	param3="${param3_values[$index]}" #cell_type index
	
	echo "EXECUTION OF PROGRAM NUMBER:$index" #Tells me which simulations are we at

	#Here I put the name of the program I want to run multiple times each time with a 	different perturbation input (to a different population)
    python MAIN_CODE_iterate.py "$param1" "$param2" "$param3"
	
done