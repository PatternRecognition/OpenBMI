function dat= proc_baseline(dat, ival, varargin)
%dat= proc_baseline(dat, <ival>)
%dat= proc_baseline(dat, msec, <pos>)
%dat= proc_baseline(dat, msec, <opts>)
%
% baseline correction
%
% IN   dat  - data structure of continuous or epoched data
%      ival - baseline interval [start ms, end ms], default all
%      msec - length of baseline interval in msec
%      pos  - 'beginning' (default), or 'end'
%      classwise - 0 (default)
%      trialwise - 1 (default)
%      channelwise - 0 (default) to save memory
%
% OUT  dat  - updated data structure
%
% SEE proc_medianBaseline

% bb, ida.first.fhg.de
% Matthias Treder Aug 2010: Added time-frequency data support

dim = ndims(dat.x); % 3D or 4D data


if length(varargin)==1,
  opt= struct('pos', varargin{1});
else
  opt= propertylist2struct(varargin{:});
end
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'pos', 'beginning', ...
                 'classwise', 0, ...
                 'trialwise', 1, ...
                 'channelwise', 0);
if dim<=3
  [T, nC, nE]= size(dat.x);
elseif dim==4
  [freq, T, nC, nE]= size(dat.x);
else
  error('unexpected number of dimensions in dat.x');
end
nCE= nC*nE;  
  
if ~exist('ival', 'var') | isempty(ival),
  Ti= 1:size(dat.x,1);
  if isfield(dat, 't') & opt.trialwise,
    dat.refIval= dat.t([1 end]);
  end
else
  if length(ival)==1,
    msec= ival;
    switch(lower(opt.pos)),
     case 'beginning',
      dat.refIval= dat.t(1) + [0 msec];
      Ti= getIvalIndices(dat.refIval, dat);
     case 'end',
      dat.refIval= dat.t(end) - [msec 0];
      Ti= getIvalIndices(dat.refIval, dat);
     case 'beginning_exact', % 150 msec = 15 samples (not 16). 
      Ti= 1:round(msec/1000*dat.fs);
      dat.refIval= [dat.t(Ti(1)) dat.t(Ti(end))];      
     case 'end_exact',
      Ti= T+[-round(msec/1000*dat.fs)+1:0];
      dat.refIval= [dat.t(Ti(1)) dat.t(Ti(end))];
     otherwise
      error('unknown position indicator');
    end
  else
    switch(lower(opt.pos)),
     case 'beginning_exact', % [-150 0] = 15 samples not 16 (at fs= 100 Hz)
      len= round(diff(ival)/1000*dat.fs);
      Ti= getIvalIndices(ival, dat);
      Ti= Ti(1:len);
     case 'end_exact',
      len= round(diff(ival)/1000*dat.fs);
      Ti= getIvalIndices(ival, dat);
      Ti= Ti(end-len+1:end);
     otherwise,
      Ti= getIvalIndices(ival, dat);
    end
    dat.refIval= dat.t(Ti([1 end]));
  end
end

%% for dim==5, opt.channelwise is not yet implemented
if opt.classwise,
  if opt.trialwise & ~isdefault.trialwise,
    error('you cannot use both, classwise and trialwise');
  end
  if dim<=3
    for ci= 1:size(dat.y,1),
      idx= find(dat.y(ci,:));
      baseline= nanmean(nanmean(dat.x(Ti, :, idx), 1), 3);
      if opt.channelwise,
        for ic= 1:nC,
          dat.x(:,ic,idx)= dat.x(:,ic,idx) - ...
              repmat(baseline(:, ic), [T 1 length(idx)]);
        end
      else
        dat.x(:,:,idx)= dat.x(:,:,idx) - repmat(baseline, [T 1 length(idx)]);
      end
    end
  elseif dim==4
    for ci= 1:size(dat.y,1)
      idx= find(dat.y(ci,:));
      baseline= nanmean(nanmean(dat.x(:, Ti, :, idx), 2), 4);
      dat.x(:,:,:,idx)= dat.x(:,:,:,idx) - repmat(baseline, [1 T 1 length(idx)]);
    end
  end
elseif opt.trialwise,
  if dim<=3,
    if opt.channelwise,
      for ic= 1:nCE,
        dat.x(:,ic)= dat.x(:,ic) - nanmean(dat.x(Ti,ic));
      end
    else
      baseline= nanmean(dat.x(Ti, :, :), 1);
      dat.x= dat.x - repmat(baseline, [T 1 1]);
    end
  elseif dim==4
    baseline= nanmean(dat.x(:, Ti, :, :), 2);
    dat.x= dat.x - repmat(baseline, [1 T 1 1]);
  end
else
  baseline= nanmean(nanmean(dat.x(Ti, :, :), 1), 3);
  if opt.channelwise,
    for ic= 1:nC,
      dat.x(:,ic,:)= dat.x(:,ic,:) - repmat(baseline(:, ic), [T 1 nE]);
    end
  else
    dat.x= dat.x - repmat(baseline, [T 1 nE]);
  end
end
