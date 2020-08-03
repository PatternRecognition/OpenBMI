function dat= proc_medianBaseline(dat, ival)
%dat= proc_baseline(dat, ival)
%
% baseline correction using the median instead of the mean
%
% IN   dat  - data structure of continuous or epoched data
%      ival - baseline interval [start msec, end msec], default all
%
% OUT  dat  - updated data structure
%
% SEE proc_baseline

% bb, ida.first.fhg.de


if ~exist('ival', 'var') | isempty(ival),
  Ti= 1:size(dat.x,1);
  if isfield(dat, 't'),
    dat.refIval= dat.t([1 end]);
  end
else
  Ti= getivalIndices(ival, dat);
  dat.refIval= ival;
end
baseline= median(dat.x(Ti, :, :));

T= size(dat.x, 1);
dat.x= dat.x - repmat(baseline, [T 1 1]);
