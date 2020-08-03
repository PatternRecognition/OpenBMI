function dat= proc_baseline_cw(dat, ival, pos)
%dat= proc_baseline_cw(dat, <ival>)
%dat= proc_baseline_cw(dat, msec, <pos>)
%
% baseline correction
% in contrast to proc_baseline here the baseline is subtracted
% channelwise which is slower but uses less memory
%
% IN   dat  - data structure of continuous or epoched data
%      ival - baseline interval [start ms, end ms], default all
%      msec - length of baseline interval in msec
%      pos  - 'beginning' (default), or 'end'
%
% OUT  dat  - updated data structure
%
% SEE proc_medianBaseline, proc_baseline

% bb, ida.first.fhg.de

warning('obsolete: use proc_baseline with opt.channelwise=1');
if ~exist('ival', 'var'),
  Ti= 1:size(dat.x,1);
  if isfield(dat, 't'),
    dat.refIval= dat.t([1 end]);
  end
else
  if ~exist('pos','var'), pos='beginning'; end
  if length(ival)==1,
    msec= ival;
    switch(lower(pos)),
     case 'beginning',
      ival= dat.t(1) + [0 msec];
     case 'end',
      ival= dat.t(end) - [msec 0];
     otherwise
      error('unknown position indicator');
    end
  end
  Ti= getIvalIndices(ival, dat);
  dat.refIval= ival;
end

[T, nCE]= size(dat.x);
%% this might consume too much memory:
%baseline= mean(dat.x(Ti, :, :), 1);
%dat.x= dat.x - repmat(baseline, [T 1 1]);

for ic= 1:nCE,
  dat.x(:,ic)= dat.x(:,ic) - mean(dat.x(Ti,ic));
end
