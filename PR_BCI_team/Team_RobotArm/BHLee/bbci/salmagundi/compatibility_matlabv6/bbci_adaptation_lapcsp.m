function [cls,bbci]= bbci_adaptation_lapcsp(cls, bbci, ts)
%[cls,bbci] = bbci_adaptation_lapcsp(cls, bbci, <varargin>)
%
% This function allows a supervised adaptation of an LDA classifier
% operating on Laplace + CSP band-power features.

persistent curr_mrk curr_feat window_counter running restart last_ts
persistent ts_trialstart

if ischar(ts) & strcmp(ts,'init')
  running = false;
  restart = true;
  last_ts= ts;
  return
end

if bbci.adaptation.running & (~running|isempty(running)) & restart,
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % initial case
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  [bbci.adaptation, isdefault]= ...
      set_defaults(bbci.adaptation, ...
                   'delay', 500, ...
                   'offset', 500,...
                   'mrk_start', {1,2}, ...
                   'mrk_end', [11,12,21,22,23,24,25], ...
                   'mrk_noupdate',{}, ...
                   'nlaps_per_area', 2, ...
                   'buffer', 100, ...
                   'verbose', 1);
  curr_mrk = [];
  curr_feat= zeros(size(cls(1).C.w));
  if isfield(cls(1), 'fv_buffer'),
    if bbci.adaptation.verbose,
      fprintf('[adaptation_lapcsp:] Starting with already adapted classifier\n');
    end
  else    
    idx= size(bbci.analyze.features.x, 2) + [-bbci.adaptation.buffer+1:0];
    if idx(1)<=0,
      warning('less features available than specified buffer size');
      %% Then we simple put multiple copies of features in the buffer
      idx= 1 + mod(idx-1, size(bbci.analyze.features.x, 2));
    end
    cls(1).fv_buffer= struct('x', bbci.analyze.features.x(:,idx), ...
                             'y', bbci.analyze.features.y(:,idx), ...
                             'clab', {bbci.analyze.features.clab});
    cls(1).fv_buffer.ptr= 1;
    cls(1).fv_buffer.isactive= bbci.analyze.isactive;
    if bbci.adaptation.verbose,
      fprintf('[adaptation_pmean:] Starting with fresh classifier\n');
    end
  end
  window_counter = 0;
  running = true;
  if bbci.adaptation.verbose,
    disp('[adaptation lapcsp:] Adaptation started.');
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
  [toe,timeshift]= adminMarker('query', [last_ts-ts 0]-bbci.adaptation.offset);
end
last_ts= ts;

if isempty(curr_mrk),
  % not inside update interval.
  ind_startoftrial= intersect([bbci.adaptation.mrk_start{:}], toe);
  ind_quit= intersect([bbci.adaptation.mrk_noupdate{:}], toe);
  if ~isempty(ind_startoftrial),
    if bbci.adaptation.verbose,
      fprintf('[adaptation lapcsp:] Trigger received: %s\n', vec2str(toe));
    end
    % this starts a new trial.
    curr_mrk= 1 + ~isempty(intersect(bbci.adaptation.mrk_start{2}, toe));
    toe= setdiff(toe, ind_startoftrial);
    ts_trialstart= ts;
  end
  if ~isempty(ind_quit)
    if bbci.adaptation.verbose,
      fprintf('[adaptation lapcsp:] Adaptation stopped.\n');
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
if ts-ts_trialstart >= bbci.adaptation.delay,
  fn = cls(1).fv;
  new_feat = getFeature('apply',fn,0);
  curr_feat= curr_feat + new_feat.x;
  window_counter = window_counter+1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if at the end marker: put feature into fv.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(toe)
  ind_endoftrial= intersect(bbci.adaptation.mrk_end, toe);
  if ~isempty(ind_endoftrial),
    if bbci.adaptation.verbose,
      fprintf('[adaptation lapcsp:] Endmarker: %s\n', vec2str(toe));
    end
    
    %% store actual feature into the feature buffer
    cls(1).fv_buffer.x(:,cls(1).fv_buffer.ptr)= curr_feat / window_counter;
    cls(1).fv_buffer.y(:,cls(1).fv_buffer.ptr)= ...
        double([curr_mrk==1;curr_mrk==2]);
    cls(1).fv_buffer.ptr= 1+mod(cls(1).fv_buffer.ptr, bbci.adaptation.buffer);
    
    %% reselect best Laplacian channels
    ba= bbci.adaptation;
    f_score= proc_rfisherScore(cls(1).fv_buffer);
    score= zeros(length(ba.motorarea)*ba.nlaps_per_area, 1);
    sel_idx= score;
    for ii= 1:length(ba.motorarea),
      aidx= chanind(f_score, ba.motorarea{ii});
      %[dmy, si]= sort(abs(f_score.x(aidx)), 1, 'descend');
      % for version 6
      [dmy, si]= sort(-abs(f_score.x(aidx)), 1);
      idx= (ii-1)*ba.nlaps_per_area + [1:ba.nlaps_per_area];
      score(idx)= f_score.x(aidx(si(1:ba.nlaps_per_area)));
      sel_idx(idx)= aidx(si(1:ba.nlaps_per_area));
    end
    sel_clab= strhead(f_score.clab(sel_idx));
    nLaps= length(chanind(f_score, 'not','csp*'));
    isactive= cls(1).fv_buffer.isactive;
    isactive(1:nLaps)= ismember(1:nLaps, sel_idx);
    if ~isequal(isactive, cls(1).fv_buffer.isactive),
      if ba.verbose,
        fprintf('[adaptation lapcsp:] selected Laplacian channels: ');
        for ii= 1:length(sel_clab),
          fprintf('%s (%.2f)  ', sel_clab{ii}, score(ii));
        end
        fprintf('\n');
      end
      cls(1).fv_buffer.isactive= isactive;
    end
    
    %% retrain classifier on new feature set
    fv_tmp= cls(1).fv_buffer;
    fv_tmp.x= fv_tmp.x(find(cls(1).fv_buffer.isactive),:);
    C= trainClassifier(fv_tmp, bbci.setup_opts.model);
    w_tmp= C.w;
    C.w= zeros(size(cls(1).fv_buffer.x,1), 1);
    C.w(find(cls(1).fv_buffer.isactive))= w_tmp;
    cls(1).C= C;
    if bbci.log
    	writeClassifierLog('adapt', ts, cls(1));
    end;
    save(bbci.adaptation.tmpfile, 'cls');
    curr_mrk= [];
    curr_feat = zeros(size(cls(1).C.w));
    window_counter= 0;
%    if bbci.adaptation.verbose>1,
%      fprintf('[adaptation lapcsp:] new bias: %.3f\n', cls(1).C.b);
%    end
  else
    % no marker recognized.
    if bbci.adaptation.verbose>1,
      fprintf('[adaptation lapcsp:] Marker not used in adaptation: %s\n', ...
              vec2str(toe));
    end
  end
end
