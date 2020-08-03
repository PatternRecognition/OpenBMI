function [dat,state]= online_filtafterDerivation_cat(dat, state,b, a,chans,w)
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

if exist('chans','var') & isempty(chans)
  dat2 = copyStruct(dat,dat2);
  dat2.x = dat2.x(:,chans); 
  dat2.clab = {dat2.clab{chans}};
else 
  dat2 = dat;
end
dat2 = proc_linearDerivation(dat,w);
[dat2.x,state] = filter(b,a,dat2.x(:,:),state);

dat.x = cat(2,dat.x,dat2.x);
dat.clab = {dat.clab{:},dat2.clab{:}};