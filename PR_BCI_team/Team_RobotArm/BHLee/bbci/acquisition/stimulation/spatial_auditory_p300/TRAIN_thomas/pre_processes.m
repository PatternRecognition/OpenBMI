function output = pre_processes(currentState, Lut, Dict, history, varargin),
%PRE_PROCESSES Summary of this function goes here
%   Detailed explanation goes here

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
    'test', [], ...
    'nrExtraStimuli', 0);
 
%% Do some visual stuff
if ~isfield(opt.procVar, 'handle_background'),
    opt.procVar.handle_background = stimutil_initFigure(opt);
    [opt.procVar.handle_cross, opt.procVar.handle_loc]= stimutil_fixationCrossandLocations(opt, 'angle_offset', 30);
    opt.procVar.handles_all = [opt.procVar.handle_loc opt.procVar.handle_cross(1) opt.procVar.handle_cross(2)];
    if isfield(opt, 'requireResponse') && opt.requireResponse,
      opt.procVar.response_text = text(0, .1, 'Anzahl eingeben:','color', [1 1 1], 'fontsize', 30, 'HorizontalAlignment', 'center', 'Visible', 'off');
      opt.procVar.response_field = text(0, -.1, '','color', [1 1 1], 'fontsize', 30, 'HorizontalAlignment', 'center', 'Visible', 'off');
      KbName('UnifyKeyNames');
    end
end

set(opt.procVar.handle_cross, 'Visible', 'on'); drawnow;

%% Do some auditory stuff
if ~isfield(opt, 'useSpeech') || opt.useSpeech,
    if ~isfield(opt.procVar , 'speech'),
        opt.procVar.speech = struct();
    end
    soundnames = fieldnames(opt.speech);
    for i = 1:length(soundnames),
        if ~isfield(opt.procVar, 'speechWav'),
            [sound, fs]= ...
                wavread([opt.speech_dir '/speech_' getfield(opt.speech, char(soundnames(i))) '.wav']);        
            sound(opt.speakerCount, end) = 0;
            opt.procVar.speech = setfield(opt.procVar.speech, char(soundnames(i)), sound * opt.calibrated(1,1) * .5);
        end
    end
    if isfield(opt.procVar.speech, 'trialStart'), 
        stimutil_playMultiSound(opt.procVar.speech.trialStart, opt);
        pause(1);
    end
end

%% Present stimulus
oldColor = get(opt.procVar.handle_loc(opt.targetDir), 'FaceColor');
set(opt.procVar.handle_loc(opt.targetDir), 'FaceColor', [1 0 0]);
set(opt.procVar.handle_loc, 'Visible', 'on');
drawnow;
stimutil_playMultiSound(squeeze(opt.cueStream(opt.targetDir,:,:))' * opt.calibrated(opt.targetDir,opt.targetDir), opt, 'repeat', opt.repeatTarget, 'interval', .3, 'placement', opt.targetDir);
pause(2);
set(opt.procVar.handle_loc(opt.targetDir), 'FaceColor', oldColor);
set(opt.procVar.handle_loc, 'Visible', 'off');
drawnow;

%% Countdown
if opt.countdown > 0,
    if ~isfield(opt.procVar, 'countWav'),
        for i = 1:opt.countdown,
            [sound, fs]= ...
                wavread([opt.speech_dir '/speech_' int2str(i) '.wav']);
            opt.procVar.countWav{i} = zeros([opt.speakerCount, size(sound, 1)]);
            opt.procVar.countWav{i}(1,:) = sound';
        end
    end
    stimutil_playMultiSound(opt.procVar.countWav, opt, 'interval', 0.9, 'order', 'reverse');
    pause(1);
end

%% Extra stimuli for counting purposes
if opt.nrExtraStimuli,
  extra = randperm(opt.nrExtraStimuli);
  opt.procVar.extraStimuliSeq = zeros(2,extra(1)*length(opt.speakerSelected));
  opt.procVar.extraStimuliSeq(1,:) = createSequence(extra(1), length(opt.speakerSelected))
  for i = 1:size(opt.procVar.extraStimuliSeq, 2),
    opt = stim_oddballAuditorySpatial(opt, 'cueDir', opt.procVar.extraStimuliSeq(1,i), 'sendTrigger', 0);
  end
end

%% Other stuff
opt.procVar.labels = Lut(currentState).direction;
opt.procVar.labels = rmfield(opt.procVar.labels, 'nState');
output = opt.procVar;