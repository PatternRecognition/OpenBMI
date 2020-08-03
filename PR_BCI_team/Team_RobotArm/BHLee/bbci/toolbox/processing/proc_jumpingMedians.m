function dat= proc_jumpingMedians(dat, nSamples, nMedians)
%dat= proc_jumpingMedians(dat, nSamples, <nMedians>)
%
% IN   dat      - data structure of continuous or epoched data
%      nSamples - number of samples from which the median is calculated
%      nMeds    - number of intervals from which the median is calculated
%
% OUT  dat      - updated data structure
%
% SEE proc_jumpingMeans

% bb, ida.first.fhg.de


[T, nChans, nMotos]= size(dat.x);
if ~exist('nMedians','var'), nMedians= floor(T/nSamples); end

inter= round(T-nSamples*nMedians+1:nSamples:T+1);
 
xo= zeros(nMedians, nChans, nMotos);
for s= 1:nMedians,
  Ti= inter(s):inter(s+1)-1;
  xo(s,:,:)= median(dat.x(Ti,:,:), 1);
end
dat.x= xo;

if isfield(dat, 'fs'),
  dat.fs= dat.fs/nSamples;
end
if isfield(dat, 't'),
  dat.t = mean(dat.t(reshape((T-nMedians*nSamples+1):T,nSamples,nMedians)));
end
