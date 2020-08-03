function [bbci, data]= ...
  bbci_adaptation_cspp(bbci, data, marker)
%BBCI_ADAPTATION_CSPP - reselect CSP patches by retrain and append the
%filters to fixed CSP filters. Re-train the classifier on the new features.
%
%Technique by Claudia Sannelli, see
%   Sannelli C et al, J Neural Eng, 2011
%   http://dx.doi.org/10.1088/1741-2560/8/2/025012
%
%Synopsis:
%  [BBCI, DATA]= ...
%      bbci_adaptation_cspp(BBCI, DATA, 'init', PARAMs, ...)
%  [BBCI, DATA]= ...
%      bbci_adaptation_cspp(BBCI, DATA, MARKER, FEATURE)

% 05-2011 Claudia Sannelli

if exist('marker','var') && ischar(marker) && strcmp(marker, 'init'),
  opt= propertylist2struct(bbci.adaptation.param{:});
  opt= set_defaults(opt, ...
    'ival', [], ...
    'bufferSize', 60, ...
    'mrk_start', [1 2 3],...
    'mrk_end', [11,12,21,22,23,24,25,51,52,53], ...
    'featbuffer', []);
  cspp_opt= copy_struct(bbci.analyze, ...
  'patch_selectPolicy', 'patch_score', 'nPatPerPatch', 'csp_selectPolicy', 'csp_score', ...
  'patterns', 'covPolicy', 'patch', 'patch_centers', 'patch_clab', 'require_complete_neighborhood');
  if isempty(opt.ival),
    error('you need to define the adaptaiton interval');
  end
  data.adaptation{1}.opt= opt;    
  if isfield(data.adaptation{1}, 'featbuffer'),
    bbci_log_write(data.adaptation{1}.log.fid,'# Adaptation CSPP: Starting with already adapted classifier\n');
  else
    idx= [-data.adaptation{1}.opt.bufferSize+1:0] + size(opt.featsignal.x, 3);
    if idx(1) <= 0,
      warning('less features available than specified buffer size');
      % Put multiple copies of cnt segments in the buffer
      idx= 1 + mod(idx-1, size(opt.featsignal.x, 3));
    end
    data.adaptation{1}.featbuffer= struct('x', opt.featsignal.x(:,:,idx), ...
      'y', opt.featsignal.y(:,idx), ...
      'clab', {opt.featsignal.clab}, ...
      'origClab', {bbci.analyze.features.origClab}, ...
      'csp_clab', {bbci.analyze.csp_clab}, ...
      'spat_w_csp', bbci.analyze.spat_w_csp);
    opt= rmfield(opt,  'featbuffer');
    data.adaptation{1}.opt= rmfield(data.adaptation{1}.opt,  'featbuffer');
    data.adaptation{1}.cspp_opt= cspp_opt;
  end
  data.adaptation{1}.trial_start= NaN;
  data.adaptation{1}.lastcheck= -inf;
  bbci_log_write(data.adaptation{1}.log.fid, '# Adaptation <%s> started with \n patch: %s \n patches on: %s \n ival: [%d %d] \n buffer: %d', ...
    mfilename, bbci.analyze.patch, toString(bbci.analyze.features.clab), opt.ival(1), opt.ival(2), opt.bufferSize);
  return;
end

time= data.marker.current_time;
check_ival= [data.adaptation{1}.lastcheck time];
events= bbci_apply_queryMarker(data.marker, check_ival);
data.adaptation{1}.lastcheck= time;

if ~isempty(events) && isnan(data.adaptation{1}.trial_start),
  midx= find(ismember([events.desc], data.adaptation{1}.opt.mrk_start));
  if ~isempty(midx),
    data.adaptation{1}.trial_start= events(midx(1)).time;    
    data.adaptation{1}.midx= find([events.desc] == data.adaptation{1}.opt.mrk_start);
    bbci_log_write(data.adaptation{1}.log.fid, ...
      ['# Adaptation CSPP: trial start at ' data.adaptation{1}.log.time_fmt ' with marker %d.'], ...
      data.adaptation{1}.trial_start/1000, events(midx(1)).desc);
  end
end

if isnan(data.adaptation{1}.trial_start),
  return;
end

adapt_ival= data.adaptation{1}.trial_start + data.adaptation{1}.opt.ival;
trial_adaptation_ended_by_marker= 0;

if ~isempty(events) && any(ismember([events.desc], data.adaptation{1}.opt.mrk_end));
  trial_adaptation_ended_by_marker= 1;
end
if trial_adaptation_ended_by_marker && time >= adapt_ival(2),
  
  trial_data= bbci_apply_getSegment(data.buffer, data.adaptation{1}.trial_start, data.adaptation{1}.opt.ival);

  data.adaptation{1}.featsignal.x(:,:,1:end-1)= data.adaptation{1}.featsignal.x(:,:,2:end);
  data.adaptation{1}.featsignal.y(:,1:end-1)= data.adaptation{1}.featsignal.y(:,2:end);
  
  data.adaptation{1}.featsignal.x(:,:,end)= trial_data.x;
  data.adaptation{1}.featsignal.y(:,end)= double([data.adaptation{1}.midx==1; data.adaptation{1}.midx==2]);
  
  %% reselect Patches
  [fv, spat_w_patch, score]= proc_cspp_auto(data.adaptation{1}.featbuffer, data.adaptation{1}.cspp_opt);    

  data.adaptation{1}.featsignal.spat_w_patch= spat_w_patch;
  data.adaptation{1}.featsignal.spat_w= cat(2,spat_w_patch, data.adaptation{1}.featsignal.spat_w_csp);

  linDerClab = cat(2, fv.clab, data.adaptation{1}.featsignal.csp_clab);
  
  %% retrain classifier on new feature set
  fv= proc_linearDerivation(data.adaptation{1}.featbuffer, data.adaptation{1}.featsignal.spat_w, ...
    'clab', linDerClab);
  fv= proc_variance(fv);
  fv= proc_logarithm(fv);
  fv= proc_flaten(fv);
  
  %% update cls and feature
  bbci.classifier.C= trainClassifier(fv, bbci.setup_opts.model);
  bbci.feature.param{1}= {data.adaptation{1}.featsignal.spat_w, linDerClab};
    
  if ~isequal(fv.clab, bbci.analyze.features.clab),
    bbci_log_write(data.adaptation{1}.log.fid, '# Adaptation CSPP: selected patches: %s with scores %s', ...
      toString(fv.clab), toString(score));    
  end
  
  clear fv linDerClab
  data.adaptation{1}.trial_start= NaN;
end

