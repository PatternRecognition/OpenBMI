function [cls,bbci]= bbci_adaptation_ensemble(cls, bbci, ts)
%[cls,bbci] = bbci_adaptation_ensemble(cls, bbci, <varargin>)
%
% This function allows an unsupervised adaptation of the bias of LDA.
% It is called by adaptation.m, which is called by bbci_bet_apply.m
%
% bbci.adaptation options specific for adaptation_ensemble:
%   .verbose - 0: no output, 1: little output, 2: each bias change is
%      reported; default: 1
% bbci.adaptation should have the following fields for this to work:
%   .running - if this is 1, the adaptation process is going on.

persistent curr_mrk curr_feat window_counter running restart last_ts
persistent ts_trialstart trial_no outputs outputs_all loss old_event
persistent out label N adapt_w_on C0 loss_cl1 loss_cl2

if ischar(ts) & strcmp(ts,'init')
  running = false;
  restart = true;
  last_ts= ts;
  trial_no=1;
  outputs=[];
  outputs_all=[];
  loss=[];
  N.cl1=[];
  N.cl2=[];
  N.adapt=5;
  adapt_w_on=[];
  C0=cls.C;
  loss_cl1=[];
  loss_cl2=[];
  return
end

if bbci.adaptation.running & (~running|isempty(running)) & restart,
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % initial case
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  [bbci.adaptation, isdefault]= ...
      set_defaults(bbci.adaptation, ...
                   'UC', 0.05,...
                   'delay', 250, ...
                   'offset', 500,...
                   'mrk_start', [1 2],...
                   'mrk_end', [11,12,21,22,23,24,25,51,52,53], ...
                   'mrk_noupdate',{},...
                   'load_tmp_classifier', 0, ...
                   'verbose', 1);
  if iscell(bbci.adaptation.mrk_start),
    warning('mrk_start should be a vector, not a cell array');
    bbci.adaptation.mrk_start= [bbci.adaptation.mrk_start{:}];
  end
  curr_mrk = [];
  curr_feat= zeros(size(cls(1).C.w));

  window_counter = 0;
  running = true;
  if bbci.adaptation.verbose,
    disp('[adaptation_ensemble:] Adaptation started.');
    fprintf('temp classifier will be saved as %s\n', bbci.adaptation.tmpfile);
  end
  fprintf('\n[adaptation_pmean:] adaptation parameters:\n');
  bbci.adaptation
  fprintf('\n');
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
  [toe,timeshift]= adminMarker('query', [last_ts-ts 0]-bbci.adaptation.offset);
end
last_ts= ts;

if isempty(curr_mrk),
  % not inside update interval.
  ind_startoftrial = intersect(bbci.adaptation.mrk_start, toe);
  ind_quit = intersect([bbci.adaptation.mrk_noupdate{:}], toe);
  if ~isempty(ind_startoftrial),
    if bbci.adaptation.verbose,
      fprintf('[adaptation_ensemble:] Trigger received: %s\n', vec2str(toe));
    end
    % this starts a new trial.
    curr_mrk= 1;
%    find_ind= find(ind_startoftrial(1)==toe);
%    store_ts= ts + timeshift(find_ind);
old_event=toe;    
toe= setdiff(toe, ind_startoftrial);
    ts_trialstart= ts;
  end
  if ~isempty(ind_quit)
    if bbci.adaptation.verbose,
      fprintf('[adaptation_ensemble:] Adaptation stopped.\n');
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
      fprintf('[adaptation_ensemble:] Endmarker: %s  (wc: %d)\n', vec2str(toe), window_counter);
    end
    if window_counter>0,
      curr_feat= curr_feat/window_counter; 
      % check if we should adapt:
      label(trial_no)=(old_event-1.5)*2;
      if label(trial_no)==-1,N.cl1=[N.cl1 trial_no];,end
       if label(trial_no)==1,N.cl2=[N.cl2 trial_no];,end
     
      if length(N.cl1)>=N.adapt && length(N.cl2)>=N.adapt
	adapt_w_on=1;
      else
	adapt_w_on=0;
      end
      [out0 out0_all]=apply_separatingHyperplaneEAdapt(C0,curr_feat);
      outputs_all(:,trial_no)=out0_all;
      
      if adapt_w_on
	% use only the last N.adapt outputs per class:
	inx= [N.cl1(end-N.adapt+1:end) N.cl2(end-N.adapt+1:end)];
	bias=outputs_all(:,inx);
	cls.C.bias= 0.99*cls.C.bias - 0.01*mean(bias,2);
	%cls.C.z=0.99*cls.C.z
      end
      [out out_all]=apply_separatingHyperplaneEAdapt(cls.C,curr_feat);
     
      loss1=sign(out_all)==(old_event-1.5)*2;
      loss=cat(2,loss,loss1);
      size(loss);
      [val,inx]=sort(sum(loss,2),'descend');
      n=sum(abs(val)>=floor(0.7*trial_no));
      
      	
      % class related loss:
      if  label(trial_no)==-1
	length(N.cl1);
	loss_cl1(length(N.cl1))=sign(out)~=-1; 
      elseif  label(trial_no)==1
	length(N.cl2);
      loss_cl2(length(N.cl2))=sign(out)~=1;
      end
      
      
      fprintf('|raw %3.1f|N %i|bias %2.2f|mean_out %3.1f|loss_cl1 %3.1f|loss_cl2 %3.1f \n', ...
	      out0, ...
	      n, ...
	      mean(cls.C.bias), ...
	      mean(out_all), ...
	      100*mean(loss_cl1), ...
	      100*mean(loss_cl2))
    
        if trial_no==200
 	asd
       end
       trial_no=trial_no+1;
       %writeClassifierLog('adapt', ts, cls(1));
       %save(bbci.adaptation.tmpfile, 'cls');
    end
    
    curr_mrk= [];
    curr_feat = zeros(size(cls(1).C.w));
    window_counter= 0;
    if bbci.adaptation.verbose,
      fprintf('[adaptation_ensemble:] new bias: %.3f\n', mean(cls(1).C.bias));
    end
  else
    % no marker recognized.
    if bbci.adaptation.verbose>1,
      fprintf('[adaptation_ensemble:] Marker not used in adaptation: %s\n', ...
              vec2str(toe));
     end
  end
end
