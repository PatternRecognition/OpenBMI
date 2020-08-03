function [cls, bbci, feature]= bbci_adaptation_cspp(cls, varargin)
%[cls,bbci] = bbci_adaptation_patchcsp(cls, bbci, <varargin>)
%
% This function allows a supervised adaptation of an LDA classifier
% operating on CSPpatches + CSP band-power features.

persistent curr_mrk window_counter running restart last_ts end_of_adaptation_marker
persistent ts_trialstart

switch(length(varargin)),
 case 2,
  [bbci, ts]= deal(varargin{:});
 case 3, 
  [bbci, feature, ts]= deal(varargin{:});
end

if ischar(ts) & strcmp(ts,'init')
  running = false;
  restart = true;
  last_ts= ts;
  feature = [];
  return
end

if bbci.adaptation.running && (~running || isempty(running)) && restart,
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % initial case
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  [bbci.adaptation, isdefault]= ...
    set_defaults(bbci.adaptation, ...
    'delay', 250, ...
    'offset', 500,...
    'adaptation_ival', [], ...
    'mrk_start', {1,2}, ...
    'mrk_end', [11,12,21,22,23,24,25], ...
    'mrk_noupdate',{}, ...
    'nPat', 3, ...
    'patch_score', 'medianvar', ...
    'patch_selectPolicy', 'directorscut', ...
    'nPatPerPatch', 1, ...
    'csp_score', 'medianvar', ...
    'csp_selectPolicy', 'directorscut', ...    
    'verbose', 1,...
    'load_tmp_classifier', 0);
  
  if isempty(bbci.adaptation.adaptation_ival),
    warning('use should use ''adaptation_ival''');
    bbci.adaptation.adaptation_ival= bbci.adaptation_offset + [bbci.adaptation_delay 0];
  else
    bbci.adaptation.delay= 0;
    bbci.adaptation.offset= 0;
  end
  curr_mrk = [];
  end_of_adaptation_marker= [];
  if bbci.adaptation.load_tmp_classifier,
    if exist([bbci.adaptation.tmpfile '.mat'], 'file'),
      load(bbci.adaptation.tmpfile, 'cls');
      tmpd= dir([bbci.adaptation.tmpfile '.mat']);
      if bbci.adaptation.verbose,
        fprintf('[adaptation_patchcsp:] classifier loaded from %s with date %s\n', ...
                bbci.adaptation.tmpfile, tmpd.date);
      end
    else
      if bbci.adaptation.verbose,
        fprintf('[adaptation_patchcsp:] tmp classifier not found: %s\n', bbci.adaptation.tmpfile);
      end
    end  
  end

  if isfield(cls(1), 'fv_buffer'),
    if bbci.adaptation.verbose,
      fprintf('[adaptation patchcsp:] Starting with already adapted classifier\n');
    end
  else 
    idx = bbci.adaptation.featbuffer.ptr + [-bbci.adaptation.featbuffer.buffer+1:0];    
    if idx(1)<=0,
      warning('less features available than specified buffer size');
      %% Then we simple put multiple copies of features in the buffer
      idx = 1 + mod(idx-1, bbci.adaptation.featbuffer.ptr);
    end
    cls(1).fv_buffer = bbci.adaptation.featbuffer;      
    cls(1).fv_buffer.origClab = bbci.analyze.clab;
%     cls(1).fv_buffer.isactive= bbci.analyze.isactive;
    cls(1).fv_buffer.usedPatch = bbci.setup_opts.usedPatch;    
    if bbci.adaptation.verbose,
      fprintf('[adaptation patchcsp:] Starting with fresh classifier\n');
    end
  end
  window_counter = 0;
  running = true;
  if bbci.adaptation.verbose,
    disp('[adaptation patchcsp:] Adaptation started.');
  end
end

if ~bbci.adaptation.running & running,
  disp('Adaptation was stopped from the GUI')
  running = false;
end

if ~running,
  return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% later case: see if marker is in the queue.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isfield(cls(1),'offset')
  [toe,timeshift]= adminMarker('query', [last_ts-ts 0]);
  fprintf('bbci.adaptation.offset ignored!\n');  %% Just to check
else
  [toe,timeshift]= adminMarker('query', [last_ts-ts 0]);
end
last_ts= ts;

if isempty(curr_mrk),
  % not inside update interval.
  ind_startoftrial= intersect([bbci.adaptation.mrk_start{:}], toe);
  ind_quit= intersect([bbci.adaptation.mrk_noupdate{:}], toe);
  if ~isempty(ind_startoftrial),
    curr_mrk= 1 + ~isempty(intersect(bbci.adaptation.mrk_start{2}, toe));
    if bbci.adaptation.verbose,
      fprintf('[adaptation patchcsp:] Trigger received: %s -> class %d\n', ...
        vec2str(toe), curr_mrk);
    end
    % this starts a new trial.
    toe= setdiff(toe, ind_startoftrial);
    ts_trialstart= ts;
  end
  if ~isempty(ind_quit)
    if bbci.adaptation.verbose,
      fprintf('[adaptation patchcsp:] Adaptation stopped.\n');
    end
    running = false;
    restart = false;
    return
  end
  toe= [];
end

if isempty(curr_mrk)
  % not inside an update window. Just quit.
  return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if at the end marker: put feature into fv.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(toe)
  ind_endoftrial= intersect(bbci.adaptation.mrk_end, toe);
  if ~isempty(ind_endoftrial),
    end_of_adaptation_marker= ind_endoftrial;
   else
    % no marker recognized.
    if bbci.adaptation.verbose>1,
      fprintf('[adaptation patchcsp:] Marker not used in adaptation: %s\n', ...
              vec2str(toe));
    end
  end
end

if ~isempty(end_of_adaptation_marker) && ts-ts_trialstart >= bbci.adaptation.adaptation_ival(2) && ~isempty(curr_mrk)
 
  timeshift = -(ts-ts_trialstart-bbci.adaptation.adaptation_ival(2));
 
  % this should actually happen just offline when different runs are
  % concatenated
  if abs(timeshift) > 4000    
    return;
  end
  
  bufferData = storeContData('window', bbci.adaptation.bufferID, cls(1).fv_buffer.windowSize, timeshift);
  
  if bbci.adaptation.verbose,
    fprintf('[adaptation patchcsp:] Endmarker: %s\n', vec2str(toe));
  end  
  
  %% store actual feature into the feature buffer
  if cls(1).fv_buffer.ptr < cls(1).fv_buffer.buffer
    cls(1).fv_buffer.ptr = cls(1).fv_buffer.ptr+1;
  else
    cls(1).fv_buffer.y(:,1:end-1) = cls(1).fv_buffer.y(:,2:end);
    cls(1).fv_buffer.x(:,:,1:end-1) = cls(1).fv_buffer.x(:,:,2:end);
  end
  cls(1).fv_buffer.x(:,:,cls(1).fv_buffer.ptr) = bufferData;
  cls(1).fv_buffer.y(:,cls(1).fv_buffer.ptr) = double([curr_mrk==1;curr_mrk==2]);  
  
  if ~ischar(bbci.adaptation.setup_opts.covPolicy)
    cls(1).fv_buffer.R(:,:,curr_mrk) = cls(1).fv_buffer.R(:,:,curr_mrk) + cov(bufferData);
    for ic = 1:2
      patch_opt.covPolicy(:,:,ic) = cls(1).fv_buffer.R(:,:,ic)/length(cls(1).fv_buffer.y(ic,:));
    end
  end
  
  %% reselect Patches
  [fv, W, usedPatch, A, score] = proc_cspp_auto(cls(1).fv_buffer, bbci.adaptation.setup_opts);
  
  if ~isequal(usedPatch, cls(1).fv_buffer.usedPatch),
    if bbci.adaptation.verbose,
      fprintf('[adaptation csppatch:] selected patches centered in: ');
      for ii= 1:size(usedPatch,1),
        fprintf('%s (%.2f)  ', fv.clab{ii}, score(ii));
      end
      fprintf('\n');
    end
    cls(1).fv_buffer.usedPatch = usedPatch;
  end

  cls(1).fv_buffer.spat_w_patch =  W;
  cls(1).fv_buffer.spat_w =  cat(2,W,cls(1).fv_buffer.spat_w_csp);

  linDerClab = cat(2, fv.clab, cls(1).fv_buffer.csp_clab);
  
  %% retrain classifier on new feature set
  fv = proc_linearDerivation(cls(1).fv_buffer, cls(1).fv_buffer.spat_w, 'clab', linDerClab);
  fv = proc_variance(fv);
  fv = proc_logarithm(fv);
  fv = proc_flaten(fv);
  
  C = trainClassifier(fv, bbci.setup_opts.model);
    
  %% update cls and feature
  cls(1).C = C;  
  feature(1).proc_param{1} = {cls(1).fv_buffer.spat_w, linDerClab};
  getFeature('update', 1, 'proc_param', feature(1).proc_param);
  
  clear fv w Wtmp isactive
%   writeClassifierLog('adapt', ts, cls(1).C);
  save(bbci.adaptation.tmpfile, 'cls');

  curr_mrk= [];
  end_of_adaptation_marker= [];

end

