function fv = proc_linearDerivation_filter(epo,w,c)
% PROC_LINEARDERIVATION_FILTER APPLIES SPATIAL AND TEMPORAL FILTERS TO THE DATA
%
% usage:
%     fv = proc_linearDerivation_filter(epo,w,c);
%
% input:
%     epo    usual data structure
%     w      spatial filter 
%     c      temporal filter 
%     
% output:
%     fv     filtered signal
%
% Note: if you have 12 spatial filters and 3 temporal filters, all 12 spatial filters were applied, to the first 4 remaining channels, the first temporal filter is applied, to the next 4, the 2nd temporal one is applied and so on.
%
% See also: proc_csssp
% 
% Guido Dornhege, 31/09/05
% $Id: proc_linearDerivation_filter.m,v 1.1 2005/08/31 08:33:23 neuro_cvs Exp $

cl = size(c,2);
n = size(w,2)/cl;
T = size(c,1);

fv = proc_linearDerivation(epo,w);
da = permute(reshape(fv.x,[size(fv.x,1),n,cl,size(fv.x,3)]),[1 2 4 3]);

for i = 1:T
  for j = 1:cl
    da(i+1:end,:,:,j) = da(i+1:end,:,:,j)+c(i,j)*fv.x(1:end-i,1:n,:);
  end
end

fv.x = reshape(permute(da,[1 2 4 3]),size(fv.x));


