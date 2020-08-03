function [bbci_cls, data_adapt]= bbci_adaptation_pcovmean(bbci_cls, data_adapt, marker, varargin)
%BBCI_ADAPTATIOIN_PCOVMEAN - Supervised adapt class means and pooled covariance matrix of an LDA classifier
%
%Technique by Carmen Vidaurre, see
%  Viduarre C et al, J Neural Eng, 2011
%  http://dx.doi.org/10.1088/1741-2560/8/2/025009
%
%Synopsis:
%  [BBCI_CLS, DATA_ADAPT]= ...
%      bbci_adaptation_pcovmean(BBCI_CLS, DATA_ADAPT, 'init', PARAMs, ...)
%  [BBCI_CLS, DATA_ADAPT]= ...
%      bbci_adaptation_pcovmean(BBCI_CLS, DATA_ADAPT, MARKER, FEATURE)
%
%This function is called internally by bbci_apply_adaptation.
%
%The selectable parameters are
%  'UC_mean' - [DOUBLE, default 0.05] Update coefficient for supervised
%              adaptation of the class means.
%  'UC_pcov' - [DOUBLE, default 0.03] Update coefficient for unsupervised
%              adaptation of the pooled covariance matrix (inverse of the
%              extended ..., to be exact).
%  'ival' - [1x2 DOUBLE] Time interval relative to the start marker, during
%              which features are averaged and used for adaptation; 
%              no default - needs to be set.
%              Note: When choosing the start of the adaptation interval,
%              you have to take into account bbci.feature.ival (meaning
%              that feature are based on retrospective signals).
%  'mrk_start' - {1xnClasses CELL} defines the markers that trigger the
%              adaptation for each class; no default - needs to be set.
%  'mrk_end' - [1 nMarkers INT, default []] specifies markers that indicate
%              the end of a trial. Gathering feature information for the
%              adaptation of one trial ends, when one of those end-markers
%              is received AND the end of the adaptation interval 'ival'
%              (see above) is reached. If 'mrk_end' is empty, the first part
%              of the condition is ignored, i.e., only 'ival' is taken into
%              account.
%  'scaling' - [BOOL, default true] Defines whether the updated classifier
%              should be (re-) scaled.
%  'log_mean_limit' - [INT, default 100].

% 04-2011 Claudia Sannelli
% 01-2012 Benjamin Blankertz


if ischar(marker) && strcmp(marker, 'init'),
  opt= propertylist2struct(varargin{:});
  opt= set_defaults(opt, ...
                    'UC_mean', 0.05, ...
                    'UC_pcov', 0.03, ...
                    'ival', [], ...
                    'scaling', 1, ...
                    'mrk_start', [],...
                    'mrk_end', [], ...
                    'log_mean_limit', 100);
  if isempty(opt.ival),
    error('you need to define the adaptation interval');
  end
  if ~iscell(opt.mrk_start),
    error('you need to define .mrk_start of adaptation parameters as cell');
  end
  data_adapt.opt= opt;
  data_adapt.feature= zeros(size(bbci_cls.C.w));
  data_adapt.trial_start= NaN;
  data_adapt.lastcheck= -inf;
  bbci_log_write(data_adapt.log.fid, ...
                 '# %s started with bias=%g and options %s.', ...
                 data_adapt.opt.tag, bbci_cls.C.b, toString(opt));
  return;
else
  feature= varargin{1};
end

time= marker.current_time;
check_ival= [data_adapt.lastcheck time];
events= bbci_apply_queryMarker(marker, check_ival);
data_adapt.lastcheck= time;

if ~isempty(events) && isnan(data_adapt.trial_start),
  data_adapt.clidx= 0;
  for k= 1:length(data_adapt.opt.mrk_start),
    marker_idx= find(ismember([events.desc], data_adapt.opt.mrk_start{k}));
    if ~isempty(marker_idx),
      data_adapt.clidx= k;
      break;
    end
  end
  if data_adapt.clidx>0,
    data_adapt.trial_start= events(marker_idx(1)).time;
    data_adapt.end_marker_received= isempty(data_adapt.opt.mrk_end);
    data_adapt.counter= 0;
    bbci_log_write(data_adapt.log.fid, ...
        ['# %s at ' data_adapt.log.time_fmt ...
         ' trial started with marker %d -> class %d.'], ...
        data_adapt.opt.tag, data_adapt.trial_start/1000, ...
        events(marker_idx(1)).desc, data_adapt.clidx);
  end
end

if isnan(data_adapt.trial_start),
  return;
end

% within the adaptation interval, add up the features and count them
adapt_ival= data_adapt.trial_start + data_adapt.opt.ival;
if time >= adapt_ival(1) && time <= adapt_ival(2),
  data_adapt.feature= data_adapt.feature + feature.x(:);
  data_adapt.counter= data_adapt.counter + 1;
end

if ~isempty(events) && any(ismember([events.desc], data_adapt.opt.mrk_end));
  data_adapt.end_marker_received= 1;
end
if data_adapt.end_marker_received && time >= adapt_ival(2),
  if data_adapt.counter==0,
    return;
  end
  % handy definitions
  UCm= data_adapt.opt.UC_mean;
  UCp= data_adapt.opt.UC_pcov;
  feat= data_adapt.feature/data_adapt.counter;
  % adapt class means (supervised)
  bbci_cls.C.mean(:,data_adapt.clidx)= UCm * feat + ...
      (1 - UCm) * bbci_cls.C.mean(:,data_adapt.clidx);
  % adapt (inverse of extended) pooled covariance matrix (unsupervised)
  extfeat= [1; feat];
  v= bbci_cls.C.extinvcov*extfeat;
  bbci_cls.C.extinvcov= (1/(1 - UCp)) * ...
    (bbci_cls.C.extinvcov - UCp/(1 - UCp + UCp * extfeat' * v) * v*v');
  bbci_cls.C.extinvcov= 0.5 * (bbci_cls.C.extinvcov + bbci_cls.C.extinvcov');
  % recalculate classifier
  diff_mean= diff(bbci_cls.C.mean, 1, 2);
  bbci_cls.C.w= bbci_cls.C.extinvcov(2:end, 2:end) * diff_mean;
                
  % rescale projection vector
  if data_adapt.opt.scaling,
    bbci_cls.C.w= bbci_cls.C.w/(bbci_cls.C.w' * diff_mean)*2;
  end
  bbci_cls.C.b= -bbci_cls.C.w' * mean(bbci_cls.C.mean, 2);
  if numel(bbci_cls.C.mean) < data_adapt.opt.log_mean_limit,
    bbci_log_write(data_adapt.log.fid, ...
                   ['# %s at ' data_adapt.log.time_fmt ...
                   ' bias adapted to %g, mean adapted to %s' ...
                   ' with %d features.'], ...
                   data_adapt.opt.tag, marker.current_time/1000, ...
                   bbci_cls.C.b, toString(bbci_cls.C.mean), ...
                   data_adapt.counter);
  else
    bbci_log_write(data_adapt.log.fid, ...
                   ['# %s at ' data_adapt.log.time_fmt ...
                   ' bias adapted to %g with %d features.'], ...
                   data_adapt.opt.tag, marker.current_time/1000, ...
                   bbci_cls.C.b, data_adapt.counter);
  end
  data_adapt.feature(:)= 0;
  data_adapt.trial_start= NaN;
end
