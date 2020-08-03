function dat= proc_filterbank(dat, filt_b, filt_a)
%dat= proc_filterbank(dat, filt_b, filt_a)
%
% apply a bank of digital (FIR or IIR) forward filter(s)
%
% IN   dat    - data structure of continuous data
%      b, a   - cell array of filter coefficients as obtained by butters
%
% OUT  dat    - updated data structure
%
% SEE online_filterbank, butters

% bb, ida.first.fhg.de

if ~iscell(filt_b),
  error('should be a cell');
end

nFilters= length(filt_b);
[T, nChans, nEpochs]= size(dat.x);
nCE= nChans*nEpochs;
xo= zeros([T, nCE*nFilters]);
clab= cell(1, nChans*nFilters);
cc= 1:nCE;
for ii= 1:nFilters,
  xo(:,cc)= filter(filt_b{ii}, filt_a{ii}, dat.x(:,:));
  cc= cc + nCE;
end
dat.x= xo;
% dat.clab= clab;
