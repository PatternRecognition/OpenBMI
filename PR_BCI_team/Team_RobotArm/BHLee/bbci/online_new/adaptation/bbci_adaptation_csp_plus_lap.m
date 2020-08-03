function [CLS, data_adapt]= bbci_adaptation_csp_plus_lap(CLS, data_adapt, marker, varargin)
%BBCI_ADAPTATIOIN_CSP_PLUS_LAP - Reselect Laplacian channels (additionally to CSP) and retrain classifier
%
%Technique described in
%  Vidaurre C, Sannelli C, Müller KR, Blankertz B.
%  Machine-Learning Based Co-adaptive Calibration.
%  Neural Comput, 23(3):791-816, 2011.
%  http://dx.doi.org/10.1162/NECO_a_00089
%
%Synopsis:
%  [BBCI_CLS, DATA_ADAPT]= ...
%      bbci_adaptation_csp_plus_lap(BBCI_CLS, DATA_ADAPT, 'init', PARAMs, ...)
%  [BBCI_CLS, DATA_ADAPT]= ...
%      bbci_adaptation_csp_plus_lap(BBCI_CLS, DATA_ADAPT, MARKER, FEATURE)
%
%This function is called internally by bbci_apply_adaptation.
%
%The selectable parameters are
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
%  'buffer' - [INT, deafult 80] Number of trials that are taken into account
%              for reselection Laplacians and retraining the classifer.
%  'nlaps_per_area' - [INT, deafult 2] Number of Laplacian channels to be
%              selected within each area.

% 01-2012 Benjamin Blankertz


if ischar(marker) && strcmp(marker, 'init'),
  opt= propertylist2struct(varargin{:});
  opt= set_defaults(opt, ...
                    'ival', [], ...
                    'scaling', 1, ...
                    'mrk_start', [],...
                    'mrk_end', [], ...
                    'buffer', 80, ...
                    'area', CLS.opt.area, ...
                    'nlaps_per_area', CLS.opt.nlaps_per_area);

  if isempty(opt.ival),
    error('you need to define the adaptation interval');
  end
  if ~iscell(opt.mrk_start),
    error('you need to define .mrk_start of adaptation parameters as cell');
  end
  data_adapt.opt= opt;
  data_adapt.trial_start= NaN;
  data_adapt.lastcheck= -inf;
  data_adapt.feature= zeros(size(CLS.C.w));
  % If necessary, modify the feature buffer to have the specified size
  idx= size(CLS.fv_buffer.x, 2) + [-opt.buffer+1:0];
  if idx(1)<=0,
    warning('less features available than specified buffer size');
    % Then we simple put multiple copies of features in the buffer
    idx= 1 + mod(idx-1, size(CLS.fv_buffer.x, 2));
  end
  CLS.fv_buffer.x= CLS.fv_buffer.x(:,idx);
  CLS.fv_buffer.y= CLS.fv_buffer.y(:,idx);
  CLS.fv_buffer.ptr= 1;
  bbci_log_write(data_adapt.log.fid, ...
                 '# %s started with options %s.', ...
                 data_adapt.opt.tag, toString(opt));
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
  % Store feature (and label) of current trial into buffer
  CLS.fv_buffer.x(:,CLS.fv_buffer.ptr)= data_adapt.feature/data_adapt.counter;
  CLS.fv_buffer.y(:, CLS.fv_buffer.ptr)= 0;
  CLS.fv_buffer.y(data_adapt.clidx, CLS.fv_buffer.ptr)= 1;
  CLS.fv_buffer.ptr= 1+mod(CLS.fv_buffer.ptr, data_adapt.opt.buffer);
  % Reselect Laplacian channels
  OPT= data_adapt.opt;
  f_score= proc_rfisherScore(CLS.fv_buffer);
  score= zeros(1, length(OPT.area)*OPT.nlaps_per_area);
  sel_idx= score;
  for ii= 1:length(OPT.area),
    aidx= chanind(f_score, OPT.area{ii});
    [dmy, si]= sort(abs(f_score.x(aidx)), 1, 'descend');
    idx= (ii-1)*OPT.nlaps_per_area + [1:OPT.nlaps_per_area];
    score(idx)= f_score.x(aidx(si(1:OPT.nlaps_per_area)));
    sel_idx(idx)= aidx(si(1:OPT.nlaps_per_area));
  end
  idx_active= [sort(sel_idx), CLS.fv_buffer.idx_csp];
  if ~isequal(idx_active, CLS.fv_buffer.idx_active),
    msg= 'reselected Laplacians:';
    for ii= 1:length(sel_idx),
      msg= sprintf('%s %s (%.2f) ', msg, ...
                   strhead(f_score.clab{sel_idx(ii)}), score(ii));
    end
    CLS.fv_buffer.idx_active= idx_active;
  else
    msg= 'retrained classifier with same Laplacians.';
  end
  bbci_log_write(data_adapt.log.fid, ...
                 ['# %s at ' data_adapt.log.time_fmt ' %s'], ...
                 data_adapt.opt.tag, data_adapt.trial_start/1000, msg);
      
  % retrain classifier on new feature set
  fv= CLS.fv_buffer;
  fv.x= CLS.fv_buffer.x(CLS.fv_buffer.idx_active,:);
  C= trainClassifier(fv, CLS.model);
  w_tmp= C.w;
  C.w= zeros(size(CLS.fv_buffer.x,1), 1);
  C.w(CLS.fv_buffer.idx_active)= w_tmp;
  CLS.C= C;
      
  data_adapt.feature(:)= 0;
  data_adapt.trial_start= NaN;
end
