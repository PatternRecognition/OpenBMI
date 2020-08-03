function output = pre_processes(currentState, Lut, Dict, history, varargin),
%PRE_PROCESSES Summary of this function goes here
%   Detailed explanation goes here
% Start of trial: 150 = random
%                 151 = fixed
RANDOM_MARKER = 150;
FIXED_MARKER = 151;

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
    'test', [], ...
    'nrExtraStimuli', 0, ...
    'text_nrCounted', 'Anzahl eingeben', ...
    'randomTrial', 1);
 
%% Do some visual stuff
if ~isfield(opt.procVar, 'handle_background'),
    handlefigures('use','user_feedback');
    opt.procVar.handle_background = stimutil_initFigure(opt);
    [opt.procVar.handle_cross, opt.procVar.handle_loc]= stimutil_fixationCrossandLocations(opt, 'angle_offset', 30);
    opt.procVar.handles_all = [opt.procVar.handle_loc opt.procVar.handle_cross(1) opt.procVar.handle_cross(2)];
    if isfield(opt, 'requireResponse') && opt.requireResponse,
      opt.procVar.response_text = text(0, .1, opt.text_nrCounted,'color', [1 1 1], 'fontsize', 30, 'HorizontalAlignment', 'center', 'Visible', 'off');
      opt.procVar.response_field = text(0, -.1, '','color', [1 1 1], 'fontsize', 30, 'HorizontalAlignment', 'center', 'Visible', 'off');
      KbName('UnifyKeyNames');
    end
end

if ~isfield(opt.procVar, 'randomTrial'), opt.procVar.randomTrial = opt.randomTrial; end

if opt.procVar.randomTrial,
    ppTrigger(RANDOM_MARKER);
else
    ppTrigger(FIXED_MARKER);
end
    
    
set(opt.procVar.handles_all, 'Visible', 'on');
pause(opt.pre_target_pause);

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
stimutil_playMultiSound(squeeze(opt.cueLongStream(opt.targetDir,:,:))' * opt.calibrated(opt.targetDir,opt.targetDir), opt, 'repeat', opt.repeatLongTarget, 'interval', .3, 'placement', opt.targetDir);
pause(.3);
stimutil_playMultiSound(squeeze(opt.cueStream(opt.targetDir,:,:))' * opt.calibrated(opt.targetDir,opt.targetDir), opt, 'repeat', opt.repeatShortTarget, 'interval', .3, 'placement', opt.targetDir);
pause(opt.post_target_pause);
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

%% Generate trial sequence
baseSequence = randperm(length(opt.speakerSelected));
if opt.procVar.randomTrial,
    opt.procVar.trialSequence = [];
else
    opt.procVar.trialSequence = repmat(baseSequence, 1, opt.maxRounds);
end

        if isfield(opt.procVar, 'randomTrial'),
            if opt.procVar.randomTrial,
                
            else
                
            end
        end

%% Extra stimuli for counting purposes
if opt.nrExtraStimuli,
    if opt.procVar.randomTrial,
        opt.targetMarkerOffset = 10;
        opt.cueMarkerOffset = 0;
        extra = randperm(opt.nrExtraStimuli);
        opt.procVar.extraStimuliSeq = zeros(2,extra(1)*length(opt.speakerSelected));
        opt.procVar.extraStimuliSeq(1,:) = createSequence(extra(1), length(opt.speakerSelected));
    else
        opt.targetMarkerOffset = 30;
        opt.cueMarkerOffset = 20;
        extra = randperm(opt.nrExtraStimuli);
        opt.procVar.extraStimuliSeq = zeros(2,extra(1)*length(opt.speakerSelected));
        opt.procVar.extraStimuliSeq(1,:) = repmat(baseSequence, 1, extra(1));
    end
    for i = 1:size(opt.procVar.extraStimuliSeq, 2),
            opt = stim_oddballAuditorySpatial(opt, 'cueDir', opt.procVar.extraStimuliSeq(1,i), 'sendTrigger', 1, 'targetDir', opt.targetDir);
    end
end

%% Other stuff
opt.procVar.labels = Lut(currentState).direction;
opt.procVar.labels = rmfield(opt.procVar.labels, 'nState');
output = opt.procVar;