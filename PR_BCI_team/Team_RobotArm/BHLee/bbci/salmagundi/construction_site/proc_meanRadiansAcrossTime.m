function epo= proc_meanRadiansAcrossTime(epo, ival, chans)
% dat= proc_meanRadiansAcrossTime(dat, <ival, chans>)
%
% calculate the average of radians-valued signals within a specified time 
% interval
%
% IN   dat   - data structure of continuous or epoched data
%      ival  - interval in which the average is to be calculated,
%              default whole time range
%      chans - cell array of channels to be selected, default all
%
% OUT  dat   - updated data structure

% bb, ida.first.fhg.de


if exist('chans','var') & ~isempty(chans),
  epo= proc_selectChannels(epo, chans);
end
if exist('ival','var') & ~isempty(ival),
  epo= proc_selectIval(epo, ival);
end

[epo.x, epo.std]= meanRadians(epo.x, 1);

if isfield(epo, 't'),
  epo.t= mean(epo.t);
end
