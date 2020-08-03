function epo= proc_pickAmplitudePeak(epo, ival, policy)
%epo= proc_pickAmplitudePeak(epo, <ival, policy>)
%
% IN   epo    - data structure of epoched data
%      ival   - extraction interval [start ms, end ms], default all
%      policy - 'min', 'max' or 'both', default 'both'
%
% OUT  epo    - updated data structure

% bb 03/03, ida.first.fhg.de


if ~exist('policy','var') | isempty(policy),
  policy= 'both';
end

if exist('ival','var') & ~isempty(ival),
  epo= proc_selectIval(epo, ival);
end

switch(policy)
 case {1, 'min'},
  epo.x= min(epo.x);
 case {2, 'max'},
  epo.x= max(epo.x);
 case {3, 'both'},
  epo.x= cat(1, min(epo.x), max(epo.x));
 otherwise,
  error('policy not known');
end

if isfield(epo, 't'),
  epo.t= max(epo.t);
end
