function dat= proc_arCoefsPlusVar(dat, order, nCoefs, method)
%dat= proc_arCoefsPlusVar(dat, order, <nCoefs=order, method='aryule')
%
% calculate the first nCoefs coefficients of AR models
% (for each channel and epoch) of specified order and append the variance
%
% IN   dat    - data structure of continuous or epoched data
%      order  - order of AR models
%      nCoefs - number of coefficients to be calculated (1..order),
%               default order
%
%      method - to estimate the AR model, e.g. 'aryule' (default), 
%               'arburg', 'arcov', 'armcov'
%
% OUT  dat    - updated data structure
%
% SEE proc_rcCoefsPlusVar, proc_arCoefs

% bb, ida.first.fhg.de


if ~exist('nCoefs', 'var'), nCoefs=order; end
if ~exist('method','var'), method='aryule'; end

[T, nChans, nEvents]= size(dat.x);
x_ar= zeros(order+1, nChans, nEvents);

for ce= 1:nChans*nEvents,
  ar= feval(method, dat.x(:,ce), order);
  x_ar(:,ce)= [ar(2:end)'; var(dat.x(:,ce))];
%  va(1,ce) = cov(dat.x(:,ce)');
end

dat.x= x_ar;
dat.t= dat.t(end);
