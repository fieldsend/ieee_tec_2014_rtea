function [ legal ] = check_dtlz_legality(x, func_arg )


% checks whether x is in legal limits of DTLZ domain
% retirn 1 if inside box constraints, 0 otherwise
%
% Author: Jonathan Fieldsend, University of Exeter, 2014

legal =0;
up = sum(x>func_arg.upb);
lw = sum(x<func_arg.lwb); % check in legal range

if (up+lw)==0
    legal = 1;
end


