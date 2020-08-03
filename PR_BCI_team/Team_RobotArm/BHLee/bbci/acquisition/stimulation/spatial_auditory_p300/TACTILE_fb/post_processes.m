function [newState, selection, output] = post_processes(clOut, currentState, Lut, history, varargin);
%POST_PROCESSES Summary of this function goes here
%   Detailed explanation goes here

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
  'test', [],...
  'trial_pause', 5, ...
  'requireResponse', 0, ...
  'responseOffset', 100, ...
  'nrExtraStimuli', 0, ...
  'appPhase', 'train', ...
  'requireResponse', 1);

if isfield(opt, 'tact_duration') && opt.tact_duration,
  send_trigger_vest('send', []);
end

newState = Lut(currentState).direction(clOut.class_label).nState;
if strmatch(clOut.class_label, 'exit', 'exact'),
  selection = -10;
else
  selection = [];
end

if ~strcmp(opt.appPhase, 'train') && opt.errorPRec,
    pause(1);
    stimutil_playMultiSound({opt.cueStream(clOut.class_label,:) * opt.calibrated(clOut.class_label,clOut.class_label)}, opt, 'placement', clOut.class_label, 'cue_trigger', opt.errorPTrig+(opt.procVar.target == clOut.class_label));
    pause(1);
end

if opt.nrExtraStimuli && strcmp(opt.appPhase, 'train'),
  fprintf('Actual number was: %i\n', opt.maxRounds + length(find(opt.procVar.extraStimuliSeq == opt.procVar.target)));
end

set(opt.procVar.handle_cross, 'Visible', 'off');  

if ~strcmp(opt.appPhase, 'train') && isfield(opt.procVar, 'handle_tick'),
    if opt.procVar.target == clOut.class_label,
        set(opt.procVar.handle_tick{1}, 'visible', 'on');
        opt.procVar.scores(1) = opt.procVar.scores(1)+1;
        set(opt.procVar.handle_scores(1), 'string', num2str(opt.procVar.scores(1)));
    else
        set(opt.procVar.handle_tick{2}, 'visible', 'on');
        opt.procVar.scores(2) = opt.procVar.scores(2)+1;
        set(opt.procVar.handle_scores(2), 'string', num2str(opt.procVar.scores(2)));        
    end
    set([opt.procVar.handle_scores opt.procVar.handle_scores_lab], 'visible', 'on');
    pause(2);
    set(opt.procVar.handles_all, 'visible', 'off');
end
  
if isfield(opt, 'requireResponse') && opt.requireResponse,
  while KbCheck; end
  escapeKey = KbName('RETURN');
  backKey = KbName('BACKSPACE');
  ListenChar(2);
  set(opt.procVar.response_field, 'String', '');
  set(opt.procVar.response_text, 'Visible', 'on');
  set(opt.procVar.response_field, 'Visible', 'on');drawnow;pause(.01);
  correctInput = 0;
  while correctInput == 0;
    respInput = '';
    set(opt.procVar.response_field, 'String', respInput);drawnow;pause(.01);
    
    while 1,
      [isdown dummy, keycodedown]=KbCheck;
      if isdown,
        if keycodedown(escapeKey),
          break;
        elseif keycodedown(backKey) && ~isempty(respInput),
          respInput = respInput(1:end-1);
          set(opt.procVar.response_field, 'String', respInput);drawnow;pause(.01);
        else
          keyPressed = KbName(keycodedown);
          if ~iscell(keyPressed) && ~isempty(str2num(keyPressed(1))),
            respInput = [respInput keyPressed(1)];
            set(opt.procVar.response_field, 'String', respInput);drawnow;pause(.01);
          end
        end
        while KbCheck; end
        pause(0.01);
      end
    end
    
    respInput = str2num(respInput);

    if ~isempty(respInput),
      correctInput = 1;
    end    
  end
    if opt.nrExtraStimuli,
      trigg = opt.responseOffset + (respInput - (opt.maxRounds + length(find(opt.procVar.extraStimuliSeq == opt.procVar.target))));
    else
      trigg = opt.responseOffset + (respInput - opt.maxRounds);
    end
    ppTrigger(trigg);
    set(opt.procVar.response_field, 'Visible', 'off');
    set(opt.procVar.response_text, 'Visible', 'off');drawnow;pause(.01);
    ListenChar(0);  
end

pause(opt.trial_pause);

output = opt.procVar;