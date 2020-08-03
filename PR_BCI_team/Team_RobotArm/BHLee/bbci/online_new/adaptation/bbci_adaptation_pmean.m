function [bbci_cls, data_adapt]= ...
    bbci_adaptation_pmean(bbci_cls, data_adapt, marker, varargin)
%BBCI_ADAPTATIOIN_PMEAN - Adapt pooled mean of an LDA classifier
%
%Technique by Carmen Vidaurre, see
%  Viduarre C et al, IEEE Trans Biomed Eng, 2011
%  http://dx.doi.org/10.1109/TBME.2010.2093133
%
%Synopsis:
%  [BBCI_CLS, DATA_ADAPT]= ...
%      bbci_adaptation_pmean(BBCI_CLS, DATA_ADAPT, 'init', PARAMs, ...)
%  [BBCI_CLS, DATA_ADAPT]= ...
%      bbci_adaptation_pmean(BBCI_CLS, DATA_ADAPT, MARKER, FEATURE)
%
%This function is called internally by bbci_apply_adaptation.
%
%The selectable parameters are
%  'ival', [] - Time interval relative to the start marker, during which
%              features are averaged and used for adaptation; 
%              no default - needs to be set.
%              Note: When choosing the start of the adaptation interval,
%              you have to take into account bbci.feature.ival (meaning
%              that feature are based on retrospective signals).
%  'mrk_start' - [1xnMarkers INT] defines the markers that trigger the
%              adaptation. For this unsupervised adaptation, class
%              affiliation does not matter; no default - needs to be set.
%  'mrk_end' - [1xnMarkers INT] specifies markers that indicate the end of the
%              trial. Gathering feature information for the adaptation of one
%              trial ends, when one of those end-markers is received AND
%              the end of the adaptation interval 'ival' (see above) is
%              reached. If 'mrk_end' is empty, the first part of the condition
%              is ignored, i.e., only 'ival' is taken into account; 
%              default is [].

% 03-2011 Benjamin Blankertz


if ischar(marker) && strcmp(marker, 'init'),
  opt= propertylist2struct(varargin{:});
  opt= set_defaults(opt, ...
                    'UC', 0.05,...
                    'ival', [], ...
                    'mrk_start', [1 2 3],...
                    'mrk_end', []);
  if isempty(opt.ival),
    error('Adaptation interval (.ival) must be defined');
  end
  if iscell(opt.mrk_start),
    error('Property .mrk_start must be a vector, not a cell');
  end
  data_adapt.opt= opt;
  data_adapt.feature= zeros(size(bbci_cls.C.w));
  data_adapt.trial_start= NaN;
  data_adapt.lastcheck= -inf;
  bbci_cls.C.pmean= mean(bbci_cls.C.mean, 2);
  bbci_log_write(data_adapt.log.fid, ...
                 '# %s started with bias=%g and options %s', ...
                 opt.tag, bbci_cls.C.b, toString(opt));
  return;
else
  feature= varargin{1};
end

time= marker.current_time;
check_ival= [data_adapt.lastcheck time];
events= bbci_apply_queryMarker(marker, check_ival);
data_adapt.lastcheck= time;

if ~isempty(events) && isnan(data_adapt.trial_start),
  midx= find(ismember([events.desc], data_adapt.opt.mrk_start));
  if ~isempty(midx),
    data_adapt.trial_start= events(midx(1)).time;
    data_adapt.end_marker_received= isempty(data_adapt.opt.mrk_end);
    data_adapt.counter= 0;
    bbci_log_write(data_adapt.log.fid, ...
                   ['# %s at ' data_adapt.log.time_fmt ...
                    ' trial started with marker %d.'], ...
                   data_adapt.opt.tag, data_adapt.trial_start/1000, ...
                   events(midx(1)).desc);
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

if ~isempty(events) && ...
      any(ismember([events.desc], data_adapt.opt.mrk_end));
  data_adapt.end_marker_received= 1;
end
if data_adapt.end_marker_received && time >= adapt_ival(2),
  if data_adapt.counter==0,
    return;
  end
  bbci_cls.C.pmean= (1-data_adapt.opt.UC) * bbci_cls.C.pmean + ...
      data_adapt.opt.UC * data_adapt.feature / data_adapt.counter;
  bbci_cls.C.b= -bbci_cls.C.w' * bbci_cls.C.pmean;
  bbci_log_write(data_adapt.log.fid, ...
                 ['# %s at ' data_adapt.log.time_fmt ...
                  ' bias adapted to %g with %d features.'], ...
                 data_adapt.opt.tag, marker.current_time/1000, ...
                 bbci_cls.C.b, data_adapt.counter);
  data_adapt.feature(:)= 0;
  data_adapt.trial_start= NaN;
end
