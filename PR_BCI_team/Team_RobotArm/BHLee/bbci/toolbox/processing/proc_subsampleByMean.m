function [dat, mrk]= proc_subsampleByMean(dat, nSamples, mrk)
%dat= proc_subsampleByMean(dat, nSamples)
%
% IN   dat      - time series
%      nSamples - number of samples from which the mean is calculated
%
% OUT  dat      - processed time series

%warning('obsolete: use proc_jumpingMeans');

dat= proc_jumpingMeans(dat, nSamples);

if nargin>2 & nargout>1,
  mrk.pos= round((mrk.pos-nSamples/2+0.5)/nSamples);
  mrk.fs= mrk.fs/nSamples;
end

