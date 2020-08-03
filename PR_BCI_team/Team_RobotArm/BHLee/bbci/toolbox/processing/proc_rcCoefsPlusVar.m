function dat= proc_rcCoefsPlusVar(dat, order, nCoefs, method)
%dat= proc_rcCoefsPlusVar(dat, order, <nCoefs=order, method='aryule')
%
% calculate the first nCoefs reflection coefficients of AR models
% (for each channel and epoch) of specified order and append the variance
%
% IN   dat    - data structure of continuous or epoched data
%      order  - order of AR models
%      nCoefs - number of coefficients to be calculated (1..order),
%               default order
%
%      method - to estimate the AR model, e.g. 'aryule' (default), 'arburg'
%
% OUT  dat    - updated data structure
%
% SEE proc_arCoefsPlusVar, proc_arCoefs

% bb, ida.first.fhg.de


if ~exist('nCoefs', 'var'), nCoefs=order; end
if ~exist('method','var'), method='aryule'; end

[T, nChans, nEvents]= size(dat.x);
x_rc= zeros(nCoefs+1, nChans, nEvents);

for ce= 1:nChans*nEvents,
  [rc,rc,rc]= feval(method, dat.x(:,ce), order);
  x_rc(:,ce)= [rc(1:nCoefs); var(dat.x(:,ce))];
end

dat.x= x_rc;
dat.t= dat.t(end);
