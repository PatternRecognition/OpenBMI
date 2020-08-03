function [cls,bbci]= bbci_adaptation_mean(cls, bbci, ts)
%[cls,bbci] = bbci_adaptation_pcovmean(cls, bbci, <varargin>)
%
% This function performs a supervised adaptation of LDA. The inverse
% of the covariance matrices are updates by the matrix inversion
% lemma.
% It is called by adaptation.m, which is called by bbci_bet_apply.m
%
% Technique see Vidaurre et al ?.
%
% bbci.adaptation options specific for adaptation_pmean:
%   .verbose - 0: no output, 1: little output, 2: each bias change is
%      reported; default: 1
% bbci.adaptation should have the following fields for this to work:
%   .running - if this is 1, the adaptation process is going on.

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
                   'UC_mean', 0.075,...
                   'delay', 250, ...
                   'offset', 500,...
                   'adaptation_ival', [], ...
                   'mrk_start', {1,2},...
                   'mrk_end', [11,12,21,22,23,24,25], ...
                   'mrk_noupdate',{},...
                   'scaling',1,...
                   'load_tmp_classifier', 0, ...
                   'verbose', 1);
  
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
  window_counter = 0;
  running = true;
  if ~iscell(bbci.adaptation.mrk_start),
    warning('bbci.adaptation.mrk_start should be a cell array, one cell for each class.');
    bbci.adaptation.mrk_start= num2cell(bbci.adaptation.mrk_start);
  end
  if bbci.adaptation.load_tmp_classifier,
    if exist([bbci.adaptation.tmpfile '.mat'], 'file'),
      load(bbci.adaptation.tmpfile, 'cls');
      tmpd= dir([bbci.adaptation.tmpfile '.mat']);
      if bbci.adaptation.verbose,
        fprintf('[adaptation_pcovmean:] classifier loaded from %s with date %s\n', ...
                bbci.adaptation.tmpfile, tmpd.date);
      end
    else
      if bbci.adaptation.verbose,
        fprintf('[adaptation_pcovmean:] tmp classifier not found: %s\n', bbci.adaptation.tmpfile);
      end
    end  
  end
  if bbci.adaptation.verbose,
    disp('[adaptation pcovmean:] Adaptation started.');
    if bbci.adaptation.verbose>1,
      disp(bbci.adaptation);
    end
  end
end

if ~bbci.adaptation.running & running,
  disp('Adaptation was stopped from the GUI')
  running = false;
  %if isfield(bbci,'update_port') & ~isempty(bbci.update_port)
  %  send_data_udp(bbci.gui_machine,bbci.update_port,...
  %                    double(['{''bbci.adaptation.running'',0}']));
  %else
  %  warning('No update port defined!');
  %end
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
  ind_quit = intersect([bbci.adaptation.mrk_noupdate{:}], toe);
  if ~isempty(ind_startoftrial),
    % this starts a new trial.
    curr_mrk= 1 + ~isempty(intersect(bbci.adaptation.mrk_start{2}, toe));
    if bbci.adaptation.verbose,
      fprintf('[adaptation lapcsp:] Trigger received: %s -> class %d\n', ...
        vec2str(toe), curr_mrk);
    end
    toe= setdiff(toe, ind_startoftrial);
    ts_trialstart= ts;
  end
  if ~isempty(ind_quit)
    if bbci.adaptation.verbose,
      fprintf('[adaptation pcovmean:] Adaptation stopped.\n');
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
      fprintf('[adaptation pcovmean:] Marker not used in adaptation: %s\n', ...
              vec2str(toe));
    end
  end
end

if ~isempty(end_of_adaptation_marker) & ...
      ts-ts_trialstart > bbci.adaptation.adaptation_ival(2),
  if bbci.adaptation.verbose,
      fprintf('[adaptation pcovmean:] Endmarker: %s  (%d windows)\n', end_of_adaptation_marker, window_counter);
  end
  if window_counter>0,
      % adapt bias (supervised)
      curr_feat= curr_feat/window_counter;
      UC= bbci.adaptation.UC_mean;
      cls(1).C.mean(:,curr_mrk)= (1-UC) * cls(1).C.mean(:,curr_mrk) + UC * curr_feat;
      cls(1).C.w= cls(1).C.extinvcov(2:end,2:end) * diff(cls(1).C.mean, 1, 2);
      % rescale projection vector
      if bbci.adaptation.scaling,
        cls(1).C.w= cls(1).C.w/(cls(1).C.w'*diff(cls(1).C.mean, 1, 2))*2;
      end
      cls(1).C.b= -cls(1).C.w' * mean(cls(1).C.mean, 2);
      if bbci.log,
        writeClassifierLog('adapt', ts, cls(1));
      end
      save(bbci.adaptation.tmpfile, 'cls');
      if bbci.adaptation.verbose>1,
        fprintf('Updated means: %s\n', toString(cls(1).C.mean));
        fprintf('Updated classifier: %s\n', toString(cls(1).C));
      end
  end
    end_of_adaptation_marker= [];
    curr_mrk= [];
    curr_feat = zeros(size(cls(1).C.w));
    window_counter= 0;
end