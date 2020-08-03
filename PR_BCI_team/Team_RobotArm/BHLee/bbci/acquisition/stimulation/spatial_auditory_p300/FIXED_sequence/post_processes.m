function [newState, selection, output] = post_processes(clOut, currentState, Lut, history, varargin);
%POST_PROCESSES Summary of this function goes here
%   Detailed explanation goes here

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
  'test', [],...
  'trial_pause', 5, ...
  'requireResponse', 0, ...
  'responseOffset', 100, ...
  'nrExtraStimuli', 0);

newState = Lut(currentState).direction(clOut.class_label).nState;
selection = [];

if opt.nrExtraStimuli,
%   for i = 1:opt.nrExtraStimuli,
%     idx = randperm(length(opt.speakerSelected));
%     opt.procVar.extraStimuliSeq(2,i) = opt.speakerSelected(idx(1));
%     opt = stim_oddballAuditorySpatial(opt, 'cueDir', opt.procVar.extraStimuliSeq(2,i), 'sendTrigger', 0);
%   end
  fprintf('Actual number was: %i\n', opt.maxRounds + length(find(opt.procVar.extraStimuliSeq == opt.procVar.target)));  
end

if ~isfield(opt.procVar, 'end_tone'),
    opt.procVar.end_tone = stimutil_generateTone(opt.end_tone_freq, 'harmonics', 5, 'duration', opt.end_tone_duration, 'pan', [1], 'fs', 44100, 'rampon', 15, 'rampoff', 15);
end
pause(opt.post_trial_pause);
stimutil_playMultiSound(opt.procVar.end_tone * opt.calibrated(1,1), opt);

set(opt.procVar.handle_cross, 'Visible', 'off');
if ~isfield(opt, 'useSpeech') || (opt.useSpeech && isfield(opt.procVar.speech, 'relax')),
  pause(1);
  stimutil_playMultiSound(opt.procVar.speech.relax, opt);
end

if isfield(opt, 'requireResponse') && opt.requireResponse,
    while KbCheck; end
    escapeKey = KbName('RETURN');
    backKey = KbName('BACKSPACE');
    quitKey = KbName('ESCAPE');
    ListenChar(2);
    set(opt.procVar.response_field, 'String', '');
    set(opt.procVar.response_text, 'Visible', 'on');
    set(opt.procVar.response_field, 'Visible', 'on');drawnow;pause(.01);
    correctInput = 0;
    exitKey = 0;
    ignoreTrig = 0;
    while correctInput == 0;
        respInput = '';
        set(opt.procVar.response_field, 'String', respInput);drawnow;pause(.01);

        while 1,
            [isdown dummy, keycodedown]=KbCheck;
            if isdown,
                if keycodedown(escapeKey),
                    exitKey = 1;
                    break;
                elseif keycodedown(quitKey),
                    exitKey = 1;
                    ignoreTrig = 1;
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
            end
            pause(0.05);
        end

        respInput = str2num(respInput);

        if ~isempty(respInput) || ignoreTrig,
            correctInput = 1;
        end
    end

    if ~ignoreTrig,
        if opt.nrExtraStimuli,
            trigg = opt.responseOffset + (respInput - (opt.maxRounds + length(find(opt.procVar.extraStimuliSeq == opt.procVar.target))));
        else
            trigg = opt.responseOffset + (respInput - opt.maxRounds);
        end
        ppTrigger(trigg);
    end
    set(opt.procVar.response_field, 'Visible', 'off');
    set(opt.procVar.response_text, 'Visible', 'off');drawnow;pause(.01);
    ListenChar(0);
end

pause(opt.trial_pause);
opt.procVar.randomTrial = ~opt.procVar.randomTrial;

output = opt.procVar;