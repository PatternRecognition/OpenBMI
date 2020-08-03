function [dat,state]= online_filtafterDerivation(dat, state,b, a,w)
%[dat,state]= online_filtafterderivation(dat, state,b, a,w)
%
% apply digital (FIR or IIR) filter
%
% IN   dat   - data structure of continuous or epoched data
%      b, a     - filter coefficients
%      w     - spatial filter
%
% OUT  dat      - updated data structure

% bb, ida.first.fhg.de

if ~exist('w','var') | isempty(w)
  global hlp_w;
  w = hlp_w;
end


dat = proc_linearDerivation(dat,w);
[dat.x(:,:),state] = filter(b,a,dat.x(:,:),state);
