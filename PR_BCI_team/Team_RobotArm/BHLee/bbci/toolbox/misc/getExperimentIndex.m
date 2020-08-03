function ip= getExperimentIndex(expbase, paradigm, appendix, index)
%exp_idx= getExperimentIndex(expbase, paradigm, appendix, index)
%  or
%N= getExperimentIndex(expbase, paradigm, appendix)
%
% IN   expbase    - database of experiments, see readDatabase
%      paradigm   - name of paradigm, e.g. 'selfpaced'
%      appendix   - specification of paradigm, e.g. '2s', or []
%      index      - index to paradigm (leave out to get all indices)
%
% OUT  exp_idx    - index to expbase
%      N          - number of experiments with given paradigm/specification
%
% appendix=[] is used as wildcard

if ~exist('appendix','var'), appendix=[]; end;
if ~exist('index','var'), index=[]; end;

if isequal(index, 'all'),
  N= getExperimentIndex(expbase, paradigm, appendix);
  ip= getExperimentIndex(expbase, paradigm, appendix, 1:N);
  return;
end

if length(index)>1,
  ip= [];
  for ii= 1:length(index),
    ip= [ip, getExperimentIndex(expbase, paradigm, appendix, index(ii))];
  end
  return;
end

nExps= length(expbase);
ie= 0;
ip= 0;
while ie<nExps & (isempty(index) | ip<index),
  ie= ie+1;
  if strpatterncmp(paradigm, expbase(ie).paradigm) & ...
        (isempty(appendix) | strpatterncmp(appendix, expbase(ie).appendix)),
    ip= ip+1;
    if isequal(ip, index),
      ip= ie;
      break;
    end
  end
end
