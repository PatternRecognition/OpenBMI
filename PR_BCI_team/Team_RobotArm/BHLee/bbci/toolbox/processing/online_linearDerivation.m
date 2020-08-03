function [out]= online_linearDerivation(dat, A, clab)
%out= online_linearDerivation(dat, A, <clab>)

out= dat;
out.x= dat.x*A;
if nargin>=3,
  out.clab= clab;
end
