function [I_dom, X, Y_mo, Y_n, Y_dom] = RTEA_single(evaluations,initial_function,cost_function,evolution_function,num_obj,func_arg)


% function [I_dom, X, Y_mo, Y_n, Y_dom] = RTEA_single(evaluations,extra_evals,cost_function,domain_function,l,num_obj,p_mut,p_cross,func_arg)
%
% Code relates to:
% Fieldsend JE & Everson RM.
% The Rolling Tide Evolutionary Algorithm:  A Multi-Objective Optimiser 
% for Noisy Optimisation Problems, 
% IEEE Transactions on Evolutionary Computation, 
% (in press) -- available online via IEEE Explore since Feburary 2014.
%
% Please cite the above work if you use this code
%
% This modified version takes in functions to generate solutions, mutate
% and crosover, seperating problem specific requirements from the high
% level optimiser
%
% Designed for optimisation problems where there is
% observational noise -- assumes noise is symmetric with the mean being the 
% appropriate ML estimator (e.g. Guassian).
%
% Uses the single_link_guardian_iterative function, which is also available
% from my github page in the gecco_2014_changing_objectives repository
%
% INPUTS:
%
% evalutions = total number of function evaluations
% cost_function = string with name of multi-objective function to be 
%          optimised, should take a design vector and the func_arg 
%          structure as arguments and return a 1 by num_obj vector of 
%          corresponding objective values (which are probably noisy)
% num_obj = number of objectives
% func_arg = structure of cost function meta-parameters
%
% OUTPUTS:
%
% I_dom = indices of estimated Pareto set members of X and corresponding
%          Pareto front members of Y_mo
% X = All design locations evaluated by RTEA in its optimisation run
% Y_mo = (mean) objective vectors associated with the elements in X
% Y_n = number of reevaluations taken at each member of X in order to
%          generate Y_mo elements
% Y_dom = index of member of X which 'guards' each member (a value of 0
%          meaning it is not guarded, and therefore is in the estimated
%          Pareto set
%
% (c) Jonathan Fieldsend 2012, 2013, 2014, 2016

if (evaluations<1)
    error('Evaluations must be at least 1');
end
if (num_obj<1)
    error('Cannot have zero or fewer objectives');
end

search_proportion = 0.95; % use final 5% of evaluations to hone estimate
% predetermine the number of designs visited, due to the multiplication
% through by the decimal search_proportion term and the ceil this may over
% estimate by a one or two, so at the end of the run we empty the unused
% final elements of the preallocated matrices
unique_locations = 1+ceil(((evaluations-1)/2)*search_proportion);

% preallocate matrices for efficiency
X = cell(unique_locations,l); % all locations evaled
Y_mo = zeros(unique_locations,num_obj); % mean criteria values associated with solutions
Y_n = zeros(unique_locations,1); % number of revaluations of this particular solution
Y_dom = zeros(unique_locations,1); % index of set member which dominates this one.

% now sample 100 points
% propogate first sample to initialise archive
X{1} = initial_function();
Y_mo(1,:)=evaluate_f(cost_function,X{1},num_obj,1,func_arg);
Y_n(1) = 1;
I_dom = 1;
index = 2; % index of next element to add

evals = 1; % keep track of evaluations used

% OPTIMISATION LOOP
while (evals < evaluations)
    % propose a new design
 
    if (evals<=evaluations*search_proportion)
        % first 95% of evaluations for search
        % final 5% just for refinement
        
        x = evolve_new_solution(evolution_function, X, I_dom, func_arg);
    
        [X, Y_mo, Y_n, Y_dom, index, I_dom] = evaluate_and_set_state(...
            cost_function,x,num_obj,1,func_arg, X, Y_mo, Y_n, Y_dom, index, -1,I_dom,evals);
        evals = evals+1;
    end
    
    
    % update estimate of an existing elite design
    if evals<evaluations
        [~,II] = min(Y_n(I_dom));
        copy_index = I_dom(II(1));
        x = X{copy_index};
        % now see if membership of I_dom should be changed
        [X, Y_mo, Y_n,Y_dom, index, I_dom] = evaluate_and_set_state(...
            cost_function,x,num_obj,1,func_arg, X, Y_mo, Y_n,Y_dom,index, copy_index,I_dom,evals);
        evals = evals+1;
    end
    
end

% remove unused matrix elements
X{index:end} = [];
Y_mo(index:end,:) = [];
Y_dom(index:end) = [];
Y_n(index:end) = [];


%---------------
function [X, Y_mo, Y_n, Y_dom,index, I_dom] = evaluate_and_set_state(...
    cost_function,x,num_obj,initial_reevals,func_arg, X, Y_mo, Y_n, Y_dom, index, prev_index, I_dom,evals)

% index  = index of new deign
% prev_index = if a a reassessment, previous location, otherwise set at -1

[x_objectives]=evaluate_f(cost_function,x,num_obj,initial_reevals,func_arg);

if (prev_index == -1) % if it is a new design
    X{index} = x;
    Y_mo(index,:) = x_objectives;
    Y_n(index) = 1;
    % update leading edge, and maintain single link guardian data structure
    [Y_mo,Y_dom,I_dom] = single_link_guardian_iterative(Y_mo,Y_dom,I_dom,index,0,[1 2 1 2 2 2]);
else % it is a reevaluation, so incrementally update mean estimate
    Y_n(prev_index) = Y_n(prev_index)+1;
    Y_mo(prev_index,:) = Y_mo(prev_index,:) + (x_objectives-Y_mo(prev_index,:))/Y_n(prev_index);
    % update leading edge, and maintain single link guardian data structure
    [Y_mo,Y_dom,I_dom] = single_link_guardian_iterative(Y_mo,Y_dom,I_dom,index,prev_index,[1 2 1 2 2 2]);
end 

% print details every 500 evaluations
if (rem(evals,500)==0)
    fprintf('Evals %d elite %d av elite reevals %f av pop reevals %f unique designs %d\n', evals, length(I_dom), mean(Y_n(I_dom)), mean(Y_n(1:index-1)),index);
end

% update index tracking number of evaluations so far
if prev_index ==-1
    index=index+1;
end


%--------------------------------------------------------------------------
function [x, copy_index]= evolve_new_solution(evolution_function, X, I_dom,func_arg)

% randomly select two elite solutions, and use to generate child solution

I = randperm(length(I_dom));
copy_index = I_dom(I(1));
copy_index2 = copy_index;
if length(I)>1
    copy_index2 = I_dom(I(2));
end
x = X(copy_index,:);
x2 = X(copy_index2,:);
x = feval(evolution_function, x, x2, func_arg);

%--------------------------------------------------------------------------
function [results]=evaluate_f(cost_function,c,num_obj,num_reevaluations,func_arg)

% repeatedly evaluate the solution 'c' num_reevaluations times

results=zeros(num_reevaluations,num_obj); %preallocate matrix for efficiency
for i=1:num_reevaluations
    results(i,:)=feval(cost_function,c,num_obj,func_arg);
end

