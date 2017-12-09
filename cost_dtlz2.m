function [objectives]=cost_dtlz2(S,m)

% [objectives]=func_dtlz1(S,m)
% function claculates the objective valuation of solution S
% using DTLZ test function 2. 
%
% S = decision vector of real values between 0,1 (inclusive)
% m = number of objectives
% objectives = vector of objectives (costs) associated with S
% 
% Refer to - K. Deb, L. Thiele, M. Laumanns and E. Zitzler, Scalable Multi-
% Objective Optimization Test Problems, Congress on Evolutionary 
% Computation (CEC), 2002, 825—830, IEEE Press. - for problem properties
%
% number of elements in S = 10+(m-1) suggested in paper 
% (based on suggestion of k=10 )
%
% Author: Jonathan Fieldsend, University of Exeter, 3/11/09

l=length(S);
objectives=zeros(1,m);
%calculate gxM;
gxM=0;
for i=m:l;
  gxM=gxM+(S(i)-0.5)^2;
end
gxM=1+gxM;
%calculate fitness - contained in first elements of S (alongside gxM)
objectives(1)=gxM;
for i=1:m-1;
    objectives(1)=objectives(1)*cos(S(i)*pi/2);
end
for i=2:m;
  objectives(i)=gxM;
  j=1;
  while(j<=m-i)
      objectives(i)=objectives(i)*cos(S(j)*pi/2);
      j=j+1;
  end
  objectives(i)=objectives(i)*sin(S(j)*pi/2);
end