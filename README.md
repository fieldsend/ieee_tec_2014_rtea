ieee_tec_2014_rtea
==================

Matlab code for the rolling-tide evolutionary algorithm described in the IEEE Transactions on Evolutionary Computation paper: 
"The Rolling Tide Evolutionary Algorithm:  A Multi-Objective Optimiser for Noisy Optimisation Problems" 
by Jonathan E. Fieldsend and Richard M. Everson

The paper is in press and has been available online via the IEEE Explore site since Feb 2014 (issue numeber in physical copy of IEEE Transactions on Evolutionary Computation TBC).

Code additionally requires the single link guardian data structure in my gecco_2014_changing_objectives repository to run (specifically the single_link_guardian_iterative.m file)

Please run "help RTEA" in the matlab terminal for details of inputs and use.

Example usage:

```matlab
num_obj = 2; % two objectives
num_var=30; % 30 design variables
mut_width = 0.2; % mutation width
cross_prob = 0.8; % crossover probability

% set up variable ranges and bounds for use in algorithm
func_arg.range = ones(1,num_var);
func_arg.upb = ones(1,num_var);
func_arg.lwb = zeros(1,num_var);
% set up noise width to apply to decorated test function
func_arg.noise = ones(1,num_obj)*0.1;

[I_dom, X, Y_mo, Y_n, Y_dom] = ...
RTEA(10000,'cost_dtlz2_noise','check_dtlz_legality',num_var,num_obj,mut_width,cross_prob,func_arg);
```
