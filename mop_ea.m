function [P,Po,Po_mean, X,X_mo,X_n,X_p,X_var] = mop_ea(max_evaluations,cost_function,func_arg,l,num_obj)

% Implementation of the robust MOEA discribed in
% A. Syberfeldt, A. Ng, R. John, and P. Moore, 
% ?Evolutionary optimisation of noisy multi-objective problems using 
% confidence-based dynamic resampling,? 
% European Journal of Operational Research, 
% vol. 204, pp. 533?544, 2010.
%
%
% This implementation assumes all decision variables is bounded on the
% unit range -- it is the responsibility of the user to implement a
% wrapper to preform any rescaling nesessary for application to any
% problems that do not conform to this range. 
%
% Inputs
%
% max_evaluations = maximim cost_function evaluations to use in run
% obj_func = cost function to be optimised, function should expect three args: decision vector,
%           number of objectives and a structure of additional function
%           arguments
% func_arg = structure of additional arguments required by cost function
% l = number of decision variables
% num_obj = number of objectives
%
% OUTPUTS
%
% P = end population
% Po = end population objectives structure
% Po_mean = end population mean objectives
% X = all locations evaluated during run
% X_mo =  mean objectives record at locations
% X_n = number of resamples record at locations
% X_p = previous index of location record
% X_var =  variance of resamples record
%
% REQUIRES recursive_pareto_shell_with_duplicates function from the 
% emo_2013_viz repository, also on my public github page 
%
% Implementation by Jonathan Fieldsend, University of Exeter, 2013



% pop size = 50
% offspring = 25
% mutation step size = 0.5
% crossover = single point
% crossover probability = 0.8

pop_size = 50;
child_size = 25;
mut_step = 0.5;
x_over_prob = 0.8;

% r1 = 0.75 , 5
% r2 = 0.7 , 4
% r3 = 0.65 , 3
% r4 = 0.6 , 2
% r5 = 0.55 , 2
rank_alphas = ones(pop_size,1)*0.55;
rank_alphas(1:4) = [1-0.75, 1-0.7, 1-0.65, 1-0.6];
rank_n = ones(pop_size,1)*2;
rank_n(1:3) = [5, 4, 3];
n=2;


% random initial_population
P = rand(pop_size,l);
Po = []; %cell{pop_size};
Po_mean = zeros(pop_size,num_obj);
temp_fit = zeros(n,num_obj);

% record all locations sampled

X = zeros(max_evaluations,l); % location record
X_mo = zeros(max_evaluations,num_obj); % mean objectives record
X_n = zeros(max_evaluations,1); % number of resamples record
X_p = ones(max_evaluations,1)*-1; % previous index of location record
X_var = zeros(max_evaluations,num_obj); % variance of resamples record

samples_index = 1;

for i=1:pop_size
    for j=1:n
        temp_fit(j,:) = feval(cost_function,P(i,:),num_obj, func_arg);
        X(samples_index,:) = P(i,:);
        X_n(samples_index) = j;
        if j==1
            X_mo(samples_index,:) = temp_fit(j,:);
        else
            prev_index = samples_index-1;
            X_p(samples_index,:) = prev_index;
            X_mo(samples_index,:) = X_mo(prev_index,:) + (temp_fit(j,:)-X_mo(prev_index,:))/X_n(samples_index);
            X_var(samples_index,:) = var(temp_fit(1:j,:));
        end
        
        samples_index = samples_index+1;
    end
    Po(i).newest_index = samples_index-1;
    Po_mean(i,:) = mean(temp_fit);    
end

evaluations = pop_size*n;
% all initial random solutions have been evaluated n times
Parent = zeros(pop_size,l);
C = zeros(child_size,l);
Co_mean = zeros(child_size,num_obj);



print_val = 500;
while (evaluations<max_evaluations)
    
    % first select parents
    for k=1:pop_size
        % crowding tournament selection
        % to fill C
        Pr = recursive_pareto_shell_with_duplicates(Po_mean,1);
        I = randperm(pop_size);
        a = P(I(1),:);
        b = P(I(2),:);
        
        index_a = Po(I(1)).newest_index;
        index_b = Po(I(2)).newest_index;
        
        n_max_a = rank_n(Pr(I(1)));
        n_max_b = rank_n(Pr(I(2)));
        alpha = max(rank_alphas(Pr(I(1))),rank_alphas(Pr(I(1))));
        [val, extra_samples,index_a,index_b,m_a,m_b,X,X_mo,X_n,X_p,X_var,samples_index] = ...
            compare_solutions(a,b,index_a,index_b,cost_function,func_arg,n_max_a,n_max_b,num_obj,alpha,X,X_mo,X_n,X_p,X_var,samples_index);
        evaluations = evaluations + extra_samples;
        
        Po(I(1)).newest_index = index_a;
        Po(I(2)).newest_index = index_b;
        Po_mean(I(1),:) = X_mo(Po(I(1)).newest_index,:);
        Po_mean(I(2),:) = X_mo(Po(I(1)).newest_index,:);
        if (val==-1)
           Parent(k,:) = P(I(1),:);
        elseif (val==1)
           Parent(k,:) = P(I(2),:);
        else
            % else crowding distance selection
            D = calc_crowding_dist(Po_mean);
            if D(I(1)) > D(I(2)) % if I(1) is in less crowded region
                Parent(k,:) = P(I(1),:);
            else
                Parent(k,:) = P(I(2),:);
            end
        end
    end
    % now create offspring
    for k=1:child_size
        C(k,:) = single_point_crossover(Parent(k,:), Parent(k+child_size,:),x_over_prob);
        C(k,:) = mutate(C(k,:),mut_step);
        for j=1:n
            temp_fit(j,:)  =feval(cost_function,C(k,:),num_obj, func_arg);
            X(samples_index,:) = C(k,:);
            X_n(samples_index) = j;
            if j==1
                X_mo(samples_index,:) = temp_fit(j,:);
            else
                prev_index = samples_index-1;
                X_p(samples_index,:) = prev_index;
                X_mo(samples_index,:) = X_mo(prev_index,:) + (temp_fit(j,:)-X_mo(prev_index,:))/X_n(samples_index);
                X_var(samples_index,:) = var(temp_fit(1:j,:));
            end
            samples_index = samples_index+1;
        end
        Co(k).newest_index = samples_index-1;
        Co_mean(k,:) = mean(temp_fit);
    end
    evaluations = evaluations + child_size*n;
    % now update search population
    [P,Po,Po_mean] = elitist_update(P,Po,Po_mean,C,Co,Co_mean,pop_size,child_size,num_obj);
    
    if evaluations>print_val
        fprintf('Evaluations = %d\n',evaluations);
        print_val=print_val+500;
    end
   
end




%-------------------
function [Pnew,Ponew,Po_meannew] = elitist_update(P,Po,Po_mean,C,Co,Co_mean,p_s,c_s,m)


% get non-dominated subset of Co
S= zeros(c_s,1);
r2 = zeros(c_s,1);
for i=1:c_s
    r = r2;
    %S(i) = sum((sum(Co_mean<=repmat(Co_mean(i,:),c_s,1),2) == m));
    for j=1:m
        r =r + (Co_mean(:,j)<=Co_mean(i,j));
    end
    S(i) = sum(r==m); 
end
% extract just the non-dominated subset of the child pop
I = find(S==1); % just domed by itself
C = C(I,:);
Co = Co(I);
Co_mean = Co_mean(I,:);
c_s = length(I);

% now compare to current pop
D = zeros(c_s,1);
for i=1:c_s
    r=r2;
    %D(i) = sum((sum(Po_mean<=repmat(Co_mean(i,:),p_s,1),2) == m));
    for j=1:m
        r =r + (Po_mean(:,j)<=Co_mean(i,j));
    end
    D(i) = sum(r==m); 
end
II = find(D==0); % domed by no members of the Pop
if isempty(II)==0
    P = [P; C(II,:)];
    Po = [Po, Co(II)];
    Po_mean = [Po_mean; Co_mean(II,:)];
    % now need to resize
    Pr = recursive_pareto_shell_with_duplicates(Po_mean,1);
    
    Pnew=[];
    Ponew=[];
    Po_meannew=[];
    rank_index=1;
    while (size(Pnew,1) <p_s)
        old_size = size(Pnew,1); % keep track of how big before spills over
        Pnew = [Pnew; P(Pr==rank_index,:)];
        Ponew = [Ponew, Po(Pr==rank_index)];
        Po_meannew = [Po_meannew; Po_mean(Pr==rank_index,:)];
        
        distances = calc_crowding_dist(Po_meannew);
        rank_index = rank_index+1;
    end
    % fill remaining elements by sampling from the rank_index shell
    % according to crowding distances
    if (length(Ponew)>p_s)
        remove_number = size(Pnew,1)-p_s;
        % have to many, so sort based first on rank, and then on
        % distance, as we have an index into last shell added we can
        % simply focus on these though :)
        curr_pop_len = size(Pnew,1);
        
        [last_shell_distances, I_rem] = sort(distances(old_size+1:curr_pop_len));
        % I_rem gives indices of those to remove from smallest to largest
        % of the last shell added, need to shift index by previous shells
        % added though
        I_rem = I_rem + old_size;
        Pnew(I_rem(1:remove_number),:)=[];
        Ponew(I_rem(1:remove_number))=[];
        Po_meannew(I_rem(1:remove_number),:)=[];
    end
else
    % unchanged
    fprintf('No non dominated children created\n');
    Pnew=P;
    Ponew=Po;
    Po_meannew=Po_mean;
end

%-------------------
function distances = calc_crowding_dist(X)

[n,m] = size(X);
distances = zeros(n,1);
for j=1:m
   [temp,Im] = sort(X); % get index of sorted solutions on each dimension
   distances(Im(1)) = inf;
   distances(Im(end)) = inf;
   for i=2:n-1;
      distances(Im(i)) = distances(Im(i)) + (X(Im(i+1),j)-X(Im(i-1),j)); 
   end
end

%------------------- 
function c = single_point_crossover(a,b,pxo)

c=a;
if rand()<pxo
    l = length(a);
    r = randperm(l-1);
    c(r(1)+1:end)=b(r(1)+1:end);
end

%-------------------
function c = mutate(c,mut_width)

% assumes unit range on all objectives
l = length(c);
r = randperm(l);
r=r(1);
temp = c(r);
c(r) = -1;
while (c(r)<0) || (c(r)>1)
    c(r) = temp +randn()*mut_width;
end


%-------------------
function [val, extra_samples,index_a,index_b,m_a,m_b,X,X_mo,X_n,X_p,X_var,samples_index] = compare_solutions(a,b,index_a,index_b,cost_function,func_arg,n_max_a,n_max_b,num_obj,alpha,X,X_mo,X_n,X_p,X_var,samples_index)

% val = -1, a doms b
% val = 0 mutual non-dom
% val = 1, b doms a

extra_samples =0;
fin = 0;
while fin ==0
    [conf_dom,m_a,m_b,s_a,s_b,n_a,n_b,i_max] = CDR(X_n(index_a),X_n(index_b),X_mo(index_a,:),X_mo(index_b,:),X_var(index_a,:).^0.5,X_var(index_b,:).^0.5,num_obj, alpha);
    if (conf_dom==1)
        if sum(m_a<=m_b)==num_obj
            val = -1;
            fin = 1;
        else
            val = 1;
            fin = 1;
        end
    elseif ((n_a<=n_max_b) && (n_b<n_max_b)) || ((n_a<n_max_b) && (n_b<=n_max_b))
        % if still have resamples in hand
        if (s_a(i_max)>s_b(i_max)) % if width on a larger than b
            if (n_a)<n_max_a % resample a unless already at max
                results_a_new = feval(cost_function,a, num_obj,func_arg);
                [X,X_mo,X_n,X_p,X_var,samples_index] = update_sample_state(X,X_mo,X_n,X_p,X_var,samples_index,results_a_new, index_a);
                
                index_a = samples_index -1;
            else
                results_b_new = feval(cost_function,b, num_obj,func_arg);
                [X,X_mo,X_n,X_p,X_var,samples_index] = update_sample_state(X,X_mo,X_n,X_p,X_var,samples_index,results_b_new, index_b);
                
                index_b = samples_index -1;
            end
        else % width on b is larger than a
            if (n_b)<n_max_b % resample b unless already at max
                results_b_new = feval(cost_function,b, num_obj,func_arg);
                [X,X_mo,X_n,X_p,X_var,samples_index] = update_sample_state(X,X_mo,X_n,X_p,X_var,samples_index,results_b_new, index_b);
                
                index_b = samples_index -1;
            else
                results_a_new = feval(cost_function,a, num_obj,func_arg);
                [X,X_mo,X_n,X_p,X_var,samples_index] = update_sample_state(X,X_mo,X_n,X_p,X_var,samples_index,results_a_new, index_a);
                
                index_a = samples_index -1;
            end
        end
        extra_samples = extra_samples + 1;
    else
        val = 0;
        fin = 1;
    end
end

%-------------------
function [X,X_mo,X_n,X_p,X_var,samples_index] = update_sample_state(X,X_mo,X_n,X_p,X_var,samples_index,results, prev_index)

X(samples_index,:) = X(prev_index,:);
X_n(samples_index) = X_n(prev_index)+1;
X_p(samples_index,:) = prev_index;
% incremental mean and variance updates
X_mo(samples_index,:) = X_mo(prev_index,:) + (results-X_mo(prev_index,:))/X_n(samples_index);
X_var(samples_index,:) = ((X_n(samples_index)-2) * X_var(prev_index,:) + (X_n(samples_index)-1) * (X_mo(prev_index,:) - X_mo(samples_index,:)).^2  + (results - X_mo(samples_index,:)).^2  )/(X_n(samples_index)-1);


samples_index = samples_index+1;

%-------------------
function [conf_dom,mean_a,mean_b,std_a,std_b,n_a,n_b,i_max] = CDR(n_a,n_b,mean_a,mean_b,std_a,std_b,num_obj,alpha)

i_max=-1;
conf_dom=0;




% selection of confidence level
% use Welch confidence interval


mean_diff = mean_a-mean_b;

f_hat = (std_a.^2/n_a + std_b.^2/n_b) ./ ( ((std_a.^2/n_a).^2)/(n_a-1) + ((std_b.^2/n_b).^2)/(n_b-1) );
f_hat = floor(f_hat);
t_param = 1-(alpha/(2*num_obj));
mult_part = (std_a.^2/n_a + std_b.^2/n_b).^0.5;
t_val = tinv(t_param,f_hat) .* mult_part;

% if results includes 0, the cdr <=0
cdr=((mean_diff)+t_val).*((mean_diff)-t_val); 

if sum(cdr<=0)==0
    conf_dom=1;
else
    [temp,i_max] = min(cdr); %get CDR with largest range spanning 0
end



