function [objectives]=cost_dtlz2_noise(S,m,func_arg)

% decorates the DTLZ2 test problem with gaussian noise, 
% zero mean and scaled by the 'noise' vector contained in the
% func_arg structure passed in
%
% Author: Jonathan Fieldsend, University of Exeter, 2014

Objectives=randn(1,m).*func_arg.noise+cost_dtlz2(S,m);
