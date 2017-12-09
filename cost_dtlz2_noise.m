function [objectives]=cost_dtlz2_noise(S,m,func_arg)

% decorates the DTLZ2 test problem with gaussian noise, 
% zero mean and scaled by the 'noise' vector contained in the
% func_arg structure passed in
%
% S = 1 by n design vector
% m = number of objectives
% func_arg = structure of meta-parameters, must have a member
%      'noise' which is either a non-negative scalar or a 1 by m
%      vector of non-negative values  
%
% Author: Jonathan Fieldsend, University of Exeter, 2014

Objectives=randn(1,m).*func_arg.noise+cost_dtlz2(S,m);
