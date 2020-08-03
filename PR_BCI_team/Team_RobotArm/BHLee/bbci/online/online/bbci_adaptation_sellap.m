function [cls,bbci]= bbci_adaptation_selpatch(cls, bbci, ts)
%[cls,bbci] = bbci_adaptation_selpatch(cls, bbci, <varargin>)
%
% This function allows a supervised adaptation of an LDA classifier
% operating on Laplace + CSP band-power features.

persistent curr_mrk window_counter running restart last_ts end_of_adaptation_marker
persistent ts_trialstart

if ischar(ts) && strcmp(ts,'init')
  running = false;
  restart = true;
  last_ts= ts;
  return
end

patches_opt = copy_struct(bbci.adaptation, 'nPat','patch_selectPolicy', 'patch_score', 'nPatPerPatch', ...
  'csp_score', 'csp_selectPolicy');

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
    'patch_score', 'eigenvalues', ...
    'patch_selectPolicy', 'directorscut', ...
    'nPatPerPatch', 1, ...
    'csp_score', 'eigenvalues', ...
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
        fprintf('[adaptation_csp:] classifier loaded from %s with date %s\n', ...
                bbci.adaptation.tmpfile, tmpd.date);
      end
    else
      if bbci.adaptation.verbose,
        fprintf('[adaptation_selpatch:] tmp classifier not found: %s\n', bbci.adaptation.tmpfile);
      end
    end  
  end  
  if isfield(cls(1), 'fv_buffer'),
    if bbci.adaptation.verbose,
      fprintf('[adaptation selpatch:] Starting with already adapted classifier\n');
    end
  else    
    idx= size(bbci.adaptation.featbuffer.x, 3) + [-bbci.adaptation.buffer+1:0];
    if idx(1)<=0,
      warning('less features available than specified buffer size');
      %% Then we simple put multiple copies of features in the buffer
      idx= 1 + mod(idx-1, size(bbci.adaptation.featbuffer.x, 3));
    end
    cls(1).fv_buffer= struct('x', bbci.adaptation.featbuffer.x(:,:,idx), ...
      'y', bbci.adaptation.featbuffer.y(:,idx), ...            
      'spat_w', bbci.adaptation.featbuffer.spat_w, ...
      'clab', {bbci.adaptation.featbuffer.clab}, ...
      'origClab', {bbci.analyze.clab}, ...
      'patch_clab', {bbci.adaptation.featbuffer.patch_clab}, ...
      'usedPat', bbci.setup_opts.usedPat);
    cls(1).fv_buffer.ptr= size(bbci.adaptation.featbuffer.y,2);
    if bbci.adaptation.verbose,
      fprintf('[adaptation selpatch:] Starting with fresh classifier\n');
    end
  end
  window_counter = 0;
  running = true;
  if bbci.adaptation.verbose,
    disp('[adaptation selpatch:] Adaptation started.');
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
      fprintf('[adaptation selpatch:] Trigger received: %s -> class %d\n', ...
        vec2str(toe), curr_mrk);
    end
    % this starts a new trial.
    toe= setdiff(toe, ind_startoftrial);
    ts_trialstart= ts;
  end
  if ~isempty(ind_quit)
    if bbci.adaptation.verbose,
      fprintf('[adaptation selpatch:] Adaptation stopped.\n');
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
% if inside update window: average the feature 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ts-ts_trialstart >= bbci.adaptation.adaptation_ival(2) && window_counter == 0  
  timeshift = -(ts-ts_trialstart-bbci.adaptation.adaptation_ival(2));
  new_feat = getFeature('apply',bbci.adaptation.bufferID,0);
  window_counter = window_counter+1;  
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
      fprintf('[adaptation selpatch:] Marker not used in adaptation: %s\n', ...
              vec2str(toe));
    end
  end
end

if ~isempty(end_of_adaptation_marker) && ts-ts_trialstart > bbci.adaptation.adaptation_ival(2),
  if bbci.adaptation.verbose,
    fprintf('[adaptation selpatch:] Endmarker: %s  (wc: %d)\n', vec2str(toe), window_counter);
  end
  if window_counter>0,
            
    %% store actual feature into the feature buffer                   
    cls(1).fv_buffer.ptr= cls(1).fv_buffer.ptr+1;
    cls(1).fv_buffer.x(:,cls(1).fv_buffer.ptr) = new_feat;
    cls(1).fv_buffer.y(:,cls(1).fv_buffer.ptr) = double([curr_mrk==1;curr_mrk==2]); 
    
    %% reselect Patches
    [fv, W, usedPat, A, score] = proc_csppatch_auto(cls(1).fv_buffer, patches_opt);    
    
    cls(1).fv_buffer.spat_w =  W;
        
    if ~isequal(usedPat, cls(1).fv_buffer.usedPat),
%       if bbci.adaptation.verbose,
        fprintf('[adaptation patch:] selected patches centered in: ');
        for ii= 1:length(usedPat),
          fprintf('%s (%.2f)  ', fv.clab{ii}, score(ii));
        end
        fprintf('\n');
%       end
      cls(1).fv_buffer.usedPat = usedPat;      
    end
   
    %% retrain classifier on new feature set       
    fv = proc_variance(fv);
    fv = proc_logarithm(fv);    
    C = trainClassifier(fv, bbci.setup_opts.model);              
    cls(1).C = C;    
    writeClassifierLog('adapt', ts, cls(1));
    save(bbci.adaptation.tmpfile, 'cls');
  end
    
  curr_mrk= [];  
  window_counter= 0;
  end_of_adaptation_marker= [];
end
