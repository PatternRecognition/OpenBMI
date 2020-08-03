function [cls,bbci]= bbci_adaptation_schnitzel(cls,bbci,ts)
%[cls,bbci] = adaptation(cls,bbci,<varargin>)
%
% This should be used by bbci_bet_apply for an update of the 
% classifier. Only LDA hyperplane or bias can be updated.
%
% Only the first classifier can be updated so far.
% 
% bbci.adaptation should have the following fields for this to work:
%   .running - if this is 1, the adaptation process is going on.
%   .mrk_start - cell array to contain the starting markers for the 
%                trial duration. (first entry: first class - etc.)
%   .mrk_end   - cell array to contain the end markers for the 
%                trial duration. End makers are not class-specific.
%   .mrk_noupdate - cell array of markers which stop the adaptation.
%   .mrk_update- cell array of markers to re-initialize the adaptation.
%   .init      - start a new adaptation.
%   .lambda    - update parameter (0: no update; 1:complete change).
%   .min_trials- minimal amount of training trials per class.

% kraulem 08/05
persistent fv class_counter curr_mrk curr_feat window_counter running
persistent averaging free_trial_countdown restart app_feat app_out store_ts

if ischar(ts) & strcmp(ts,'init')
  running = false;
  restart = true;
  %return
end

if bbci.adaptation.running & (~running|isempty(running)) & restart,
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % initial case
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  disp('Adaptation started.');
  bbci.adaptation= ...
      set_defaults(bbci.adaptation, ...
                   'min_trials',10,...
                   'offset',500,...
                   'mrk_circle',{70},...
                   'mrk_start',{1,2},...
                   'mrk_end',{11,12,21,22,23,24,25,71},...
                   'mrk_noupdate',{},...
                   'mrk_update',{},...
                   'lambda',1,...
                   'bias_only',1,...
                   'exchange_classifier',0,...
                   'free_trials', 0);
  % prepare fv and its fields. 
  fv= struct('x',[], 'y',[]);
  % prepare curr_mrk
  curr_mrk = [];
  % prepare curr_feat
  curr_feat = zeros(size(cls(1).C.w));
  app_feat = [];
  app_out = [];
  
  % prepare class_counter
  class_counter = [0 0];
  window_counter = 0;
  running = true;
  averaging = false;
  free_trial_countdown = bbci.adaptation.free_trials;
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% later case: see if marker is in the queue.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isfield(cls(1),'offset')
        [toe,timeshift] = adminMarker('query',[-100 0]);
else
        [toe,timeshift] = adminMarker('query',[-100 0]-bbci.adaptation.offset);
end

if isempty(curr_mrk)
  % not inside update interval.
  ind1 = intersect(bbci.adaptation.mrk_start{1},toe);% possibly we need more classes.
  ind2 = intersect(bbci.adaptation.mrk_start{2},toe);
  ind3 = intersect(bbci.adaptation.mrk_circle{1},toe);
  ind4 = intersect([bbci.adaptation.mrk_noupdate{:}],toe);
  if ~isempty(ind1)|~isempty(ind2)
    fprintf('Trigger received: %i\n',toe);
    % this starts a new trial.
    curr_mrk = 1*(~isempty(ind1))+2*(~isempty(ind2));
    averaging = false;
%    uni=union(ind1,ind2);
%    find_ind=find(uni(1)==toe);
%    store_ts=ts+timeshift(find_ind);
  elseif ~isempty(ind3)
    fprintf('Trigger received: %i\n',toe);
    averaging = true;
    curr_mrk = 0;
%    find_ind=find(ind3(1)==toe);
%    store_ts=ts+timeshift(find_ind);
  end
  if ~isempty(ind4)
    fprintf('Trigger received: %i. Adaptation stopped.\n',toe);
    running = false;
    restart = false;
    return
  end
  toe = [];
end

if isempty(curr_mrk)
  % not inside an update window. Just quit.
  return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if inside update window: average the feature 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if free_trial_countdown<=0|averaging
  fn = cls(1).fv;
  new_feat = getFeature('apply',fn,0);
  curr_feat= curr_feat + new_feat.x;
  window_counter = window_counter+1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if at the end marker: put feature into fv.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(toe)
  %toe can be an array.
  ind = intersect([bbci.adaptation.mrk_end{:}],toe);
  if ~isempty(ind)
    fprintf('Endmarker: %i\n',toe);
    if free_trial_countdown<=0|averaging
      % enter the feature
      fv.x(:,end+1) = curr_feat / window_counter;
      % set the label
      fv.y(:,end+1) = double([curr_mrk==1;curr_mrk==2]);
      class_counter = sum(fv.y,2);
      % reset some values
      curr_mrk = [];
%      curr_feat = 0*curr_feat;
      curr_feat = zeros(size(cls(1).C.w));
      window_counter = 0;
    else
      free_trial_countdown = free_trial_countdown-1;
      if free_trial_countdown==0
        disp('Countdown for free_trials finished!');
      end
      curr_mrk = [];
    end
  else
    % no marker recognized.
    toe = [];
  end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if feature vector is long enough: 
% stop adaptation and calculate classifier.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~averaging
  if min(class_counter)>=bbci.adaptation.min_trials
    disp('Adapting classifier!');
    % stop everything
    curr_mrk = [];
    %bbci.adaptation.running = false;
    restart = false;
    if ~bbci.adaptation.bias_only
      % train classifier
      C = train_LDA(fv.x,fv.y);
      % normalize classifier to the same length as the old one.
      scale_old = norm(cls(1).C.w);
      scale_new = norm(C.w);
      C.w = bbci.adaptation.lambda*C.w*scale_old/scale_new + ...
            (1-bbci.adaptation.lambda)*cls(1).C.w;
      C.b = bbci.adaptation.lambda*C.b*scale_old/scale_new + ...
            (1-bbci.adaptation.lambda)*cls(1).C.b;
      % normalize again.
      C.b = C.b/norm(C.w)*scale_old;
      C.w = C.w/norm(C.w)*scale_old;
    else
      % adapt bias only - by 1D-LDA.
      out = apply_separatingHyperplane(cls(1).C,fv.x);
      C = train_LDA(out,fv.y);
      C.b = C.b/C.w;
      if C.w<0
        warning('Scaling is negative!');
      end
      if bbci.adaptation.exchange_classifier
        C.w = sign(cls(1).scale)*cls(1).C.w;
        C.b = sign(cls(1).scale)*(cls(1).C.b*(1-bbci.adaptation.lambda)...
                                  + C.b*bbci.adaptation.lambda);
      end
    end
    if bbci.adaptation.exchange_classifier
      % exchange classifier
      writeClassifierLog('adapt',ts,cls(1).C);
      cls(1).C =  C;
      % log the classifier and its new values.
      writeClassifierLog('adapt',ts,C);
    else
      % Just update the bias. Inform the GUI about this.
      cls(1).bias = C.b;
      fprintf('Bias: %f\n',cls(1).bias);  
      if isfield(bbci,'update_port') & ~isempty(bbci.update_port)
        send_data_udp(bbci.gui_machine,bbci.update_port,...
                      double(['{''cls.bias'',' num2str(cls(1).bias) '}']));
      else
        warning('No update port defined!');
      end
      % log the classifier and the new bias values.
      writeClassifierLog('adapt',ts,cls(1));
    end
    % Erase the stored feature vectors:
    fv= struct('x',[], 'y',[]);
    class_counter=[0 0];
    % Stop the adaptation:
    bbci.adaptation.running=false;
    running=false;
  end
else
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % just averaging over the cls output.
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if ~isempty(fv.x)
    disp('Averaging.');
    % stop everything
    curr_mrk = [];
    % adapt bias only - by averaging over the period.
    out = apply_separatingHyperplane(cls(1).C,fv.x);
    % Just update the bias. Inform the GUI about this.
    cls(1).bias = -mean(out);
    fprintf('Bias: %f\n',cls(1).bias);  
    if isfield(bbci,'update_port') & ~isempty(bbci.update_port)
      send_data_udp(bbci.gui_machine,bbci.update_port,...
                    double(['{''cls.bias'',' num2str(cls(1).bias) '}']));
    else
      warning('No update port defined!');
    end
    % log the classifier and the new bias values.
    writeClassifierLog('adapt',ts,cls(1));
    % reset some parameters
    fv= struct('x',[], 'y',[]);
  end
end
