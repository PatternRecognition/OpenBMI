function [cls,bbci]= bbci_adaptation_lapcsp(cls, bbci, ts)
%[cls,bbci] = bbci_adaptation_lapcsp(cls, bbci, <varargin>)
%
% This function allows a supervised adaptation of an LDA classifier
% operating on Laplace + CSP band-power features.

persistent curr_mrk curr_feat window_counter running restart last_ts end_of_adaptation_marker
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
                   'delay', 250, ...
                   'offset', 500,...
                   'adaptation_ival', [], ...
                   'mrk_start', {1,2}, ...
                   'mrk_end', [11,12,21,22,23,24,25], ...
                   'mrk_noupdate',{}, ...
                   'nlaps_per_area', 2, ...
                   'buffer', 100, ...
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
  curr_feat= zeros(size(cls(1).C.w));
  end_of_adaptation_marker= [];
  if bbci.adaptation.load_tmp_classifier,
    if exist([bbci.adaptation.tmpfile '.mat'], 'file'),
      load(bbci.adaptation.tmpfile, 'cls');
      tmpd= dir([bbci.adaptation.tmpfile '.mat']);
      if bbci.adaptation.verbose,
        fprintf('[adaptation_lapcsp:] classifier loaded from %s with date %s\n', ...
                bbci.adaptation.tmpfile, tmpd.date);
      end
    else
      if bbci.adaptation.verbose,
        fprintf('[adaptation_lapcsp:] tmp classifier not found: %s\n', bbci.adaptation.tmpfile);
      end
    end  
  end
  if isfield(cls(1), 'fv_buffer'),
    if bbci.adaptation.verbose,
      fprintf('[adaptation lapcsp:] Starting with already adapted classifier\n');
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
      fprintf('[adaptation lapcsp:] Starting with fresh classifier\n');
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
      fprintf('[adaptation lapcsp:] Trigger received: %s -> class %d\n', ...
        vec2str(toe), curr_mrk);
    end
    % this starts a new trial.
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
if ts-ts_trialstart >= bbci.adaptation.adaptation_ival(1) & ts-ts_trialstart <= bbci.adaptation.adaptation_ival(2),
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
    end_of_adaptation_marker= ind_endoftrial;
   else
    % no marker recognized.
    if bbci.adaptation.verbose>1,
      fprintf('[adaptation lapcsp:] Marker not used in adaptation: %s\n', ...
              vec2str(toe));
    end
  end
end

if ~isempty(end_of_adaptation_marker) & ...
      ts-ts_trialstart > bbci.adaptation.adaptation_ival(2),
  if bbci.adaptation.verbose,
      fprintf('[adaptation lapcsp:] Endmarker: %s  (wc: %d)\n', vec2str(toe), window_counter);
    end
    if window_counter>0,
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
        [dmy, si]= sort(abs(f_score.x(aidx)), 1, 'descend');
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
    
      writeClassifierLog('adapt', ts, cls(1));
      save(bbci.adaptation.tmpfile, 'cls');
    end
    
    curr_mrk= [];
    curr_feat = zeros(size(cls(1).C.w));
    window_counter= 0;
    end_of_adaptation_marker= [];
%    if bbci.adaptation.verbose>1,
%      fprintf('[adaptation lapcsp:] new bias: %.3f\n', cls(1).C.b);
%    end
end
