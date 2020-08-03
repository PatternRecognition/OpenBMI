function [cue_sequence, correctResponse, buttonPressed, reactionTime]= stim_tactileP300(N, trials, varargin);
%STIM_ODDBALLAUDITORY - Stimulus Presentation for Auditory Oddball with
%spatial location cues using multichannel audio output.
%
%Synopsis:
% [cue_sequence, correctResponse, buttonPressed, reactionTime]= stim_oddballAuditory(N, trials, <OPT>)
%
%Arguments:
% N: Number of stimuli per trial
% trials: Number of trials
% OPT: Struct or property/value list of optional properties:
% 'speakerCount':       Scalar: number of speaker locations
% 'speakerSelected':    Array specifying the selected locations
% 'speakerName':        Cell array 1xSpeakerCount: names of speaker
%       locations. Not yet used.
% 'isi':                Scalar: inter-stimulus interval [ms]
% 'isi_jitter':         Scalar: jitter in ISI [ms]
% 'fs':                 Scalar: PsychPortAudio requires a set sampling rate
% 'countdown':          Scalar: count down from this number before a trial
% 'targetCue':          String or Array: if string, it must be the complete
%       path to an audio file. If an array, it must contain the audio
%       stream.
% 'language':           String: multi-language presentation supported
% 'pahandle':           Scalar: PsychPortAudio identifier of the soundcard
% 'speech':             Struct: contains filenames of speechfiles (see
%       below)
%
%   SPEECH: Struct with different speech filenames
%   'targetDirection':    String: filename
%   'trialStart':         String: filename
%   'trialStop':          String: filename
%   'relax':              String: filename
%   'nextTrial':          String: filename
%   'intro':              String: filename
%
%Triggers:
%   1: FRONT stimulus
%   2: FRONT-RIGHT stimulus
%   3: RIGHT stimulus
%   4: BACK-RIGHT stimulus
%   5: BACK stimulus
%   6: BACK-LEFT stimulus
%   7: LEFT stimulus
%   8: FRONT-LEFT stimulus
% 251: beginning of relaxing period
% 252: beginning of main experiment, after countdown
% 253: end of main experiment
% 254: end

%GLOBZ: BCI_DIR, VP_CODE, SOUND_DIR

% Coded by Martijn Schreuder for spatial auditory P300 experiments
% Small adaptaion by blanker@cs.tu-berlin.de


global SOUND_DIR VP_CODE TODAY_DIR

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
    'speakerCount', 6, ...
    'speakerSelected', [1:6], ...
    'soundcard', 'M-Audio FW ASIO',...
    'fs', 44100, ...
    'countdown', 5, ...
    'language', 'english',...
    'filename', 'tactilehex_p300', ...
    'test', 0, ...
    'isi', 500,...
    'isi_jitter', 0, ...
    'delayAfterTargetPresentation', 1000, ...
    'require_response', 0, ...
    'response_markers', {'R 16', 'R  8'}, ...
    'duration_breakBetweenRounds', 3000, ...
    'resp_latency', 2000, ...
    'req_response', false, ...
    'visual_targetPresentation', 0, ...
    'visual_cuePresentation', 0, ...
    'visual_colorActive', [1 0 0], ...
    'visual_colorInactive', 0.7*[1 1 1], ...
    'bv_host', 'localhost');

% initialize visuals
opt.handle_background= stimutil_initFigure(opt);
if ~isfield(opt, 'handle_cross') || isempty(opt.handle_cross),
    [opt.handle_cross, opt.handle_loc]= stimutil_fixationCrossandLocations(opt);
end
opt.handles_all = [opt.handle_loc opt.handle_cross(1) opt.handle_cross(2)];

% initialize key if response is required
if opt.req_response
    if opt.isi <= opt.resp_latency,
        opt.isi = opt.resp_latency + 50;
        fprintf('ISI cannot be lower than the max responsetime.\n');
        fprintf('ISI is now set to: %i\n', opt.isi);
    end
    KbName('UnifyKeynames');    
    resp_button = KbName('space');    
end

if ~isfield(opt, 'cueStream'),
    error('opt must have field ''cueStream''.');
end

% if ~iscell(opt.cueStream),
%   opt.cueStream = {opt.cueStream};
% end
% 
% %Load the cue stimulus if a filename is given
% if ischar(opt.cueStream),
%     if ~isabsolutepath(opt.cueStream),
%         opt.cueStream = strcat(BCI_DIR, 'acqusition/data/sound/', opt.cueStream);
%     end
%     [opt.cueStream, inputFs]= ...
%         wavread([opt.cueStream '.wav']);
%     if ~(opt.fs == inputFS),
%         error('Sampling frequency of cue does not correspond with set sampling frequency');
%     end
% end

% Create a structure with the different cues, ea locations, stored in it.
    cueList = struct;
    for ii = 1:opt.speakerCount,
        cueList(ii).wav = zeros(opt.speakerCount, length(opt.cueStream));
        if ~opt.singleSpeaker,
            if size(opt.cueStream, 1) == 1,
                cueList(ii).wav(ii,:) = opt.cueStream;
            else
                if ismember(ii, opt.speakerSelected),
                   cueList(ii).wav(ii,:) = opt.cueStream(find(opt.speakerSelected == ii),:);
                end
            end
        else
            if size(opt.cueStream, 1) == 1,
                cueList(ii).wav(1,:) = opt.cueStream;
            else
                if ismember(ii, opt.speakerSelected),
                   cueList(ii).wav(1,:) = opt.cueStream(find(opt.speakerSelected == ii),:);
                end
            end
        end
        % use calibrated speaker gain
        cueList(ii).wav = opt.calibrated * cueList(ii).wav;
    end


% Load the different speech files for experiment guides
speech = struct;
if isfield(opt, 'speech') && ~isempty(opt.speech),
    soundNames = fieldnames(opt.speech);
    for ii = 1:length(soundNames),
        [sound, fs]= ...
            wavread([opt.speech_dir '/speech_' getfield(opt.speech, char(soundNames(ii))) '.wav']);
        if ~(fs == opt.fs),
            error('Sampling frequency of speech does not correspond with set sampling frequency');
        end
        sound = 0.6*sound'; %0.3 should be better taken care of
        % create a 8 channel sound out of a mono sound
        container = [];
        for iii = 1:opt.speakerCount,
            container = cat(1, container, sound);
        end
        % use calibrated speaker gain
        container = opt.calibrated * container;
        speech = setfield(speech, char(soundNames(ii)), container);
    end
end

% Load the different speech files for auditive countdown
for ii= 1:opt.countdown,
    [sound, fs]= ...
        wavread([opt.speech_dir '/speech_' int2str(ii) '.wav']);
    if ~(fs == opt.fs),
        error('Sampling frequency of counts does not correspond with set sampling frequency');
    end
    sound = 0.6*sound'; %0.3 should be better taken care of
    % create a 8 channel sound out of a mono sound
    container = [];
    for iii = 1:opt.speakerCount,
        container = cat(1, container, sound);
    end
    % use calibrated speaker gain
    container = opt.calibrated * container;
    count(ii).wav = container;
end

% if necessary, alter the amount of N to get a multiple of
% lenght(opt.speakerSelected)
if mod(N, length(opt.speakerSelected)) > 0,
    fprintf('Warning, number of presentations is not a multiple of the directions.\n');
    fprintf('Now using a total number of %i presentations, instead of %i\n', N-mod(N, length(opt.speakerSelected)), N);
    N = N-mod(N, length(opt.speakerSelected));
end

% set containers for response
correctResponse = zeros(trials,N);
buttonPressed = zeros(trials, N);
reactionTime = zeros(trials,N);

% create target sequence
target_sequence = [];
if trials < length(opt.speakerSelected);
    target_sequence = randperm(length(opt.speakerSelected));
    target_sequence = opt.speakerSelected(target_sequence(1:trials));
else
    for ii = 1:trials/length(opt.speakerSelected),
        target_sequence = [target_sequence opt.speakerSelected(randperm(length(opt.speakerSelected)))];
    end
    if mod(trials, length(opt.speakerSelected)) > 0
        for ii = 1:mod(trials, length(opt.speakerSelected)),
            target_sequence = [target_sequence round(rand()*(length(opt.speakerSelected)-1))+1];
        end
    end
%     target_sequence(:) = opt.speakerSelected(target_sequence(randperm(length(target_sequence))))
end

if ~isempty(opt.bv_host),
    bvr_checkparport;
end

if opt.test,
    fprintf('Warning: test option set true: EEG is not recorded!\n');
else
    if ~isempty(opt.filename),
        bvr_startrecording([opt.filename VP_CODE]);
    else
        warning('!*NOT* recording: opt.filename is empty');
    end
end

ppTrigger(251);
%present intro speech
if isfield(speech, 'intro'),
    PsychPortAudio('FillBuffer', opt.pahandle, speech.intro);
    PsychPortAudio('Start', opt.pahandle);
    PsychPortAudio('Stop', opt.pahandle, 1);
end
pause(1);
if isfield(speech, 'trialStart'),
    PsychPortAudio('FillBuffer', opt.pahandle, speech.trialStart);
    PsychPortAudio('Start', opt.pahandle);
    PsychPortAudio('Stop', opt.pahandle, 1);
end

set(opt.handle_cross, 'Visible', 'on');drawnow; 
pause(2);
if opt.req_response, 
    state= acquire_bv(1000, opt.bv_host);
end

% initiate countdown
opt.handle_msg= stimutil_initMsg;
for ii= opt.countdown:-1:1,
    fprintf('Run starts in %i seconds\n', ii);
    msg= sprintf('%d', ii);
    set(opt.handle_msg, 'String', msg, 'FontSize',0.25);
    drawnow;
    PsychPortAudio('FillBuffer', opt.pahandle, count(ii).wav);
    PsychPortAudio('Start', opt.pahandle);
    pause(1);
    PsychPortAudio('Stop', opt.pahandle, 1);
end
set(opt.handle_msg, 'Visible','off');
drawnow;
ppTrigger(252);

for zz = 1:trials,
    %**************************************************************
    % RUN <trials> amount of trials, with <N> amount of stimuli
    %**************************************************************
    % create the cue_sequence for presentation
    % create a sequence of 1xN, where each location is presented equally often.
    set(opt.handle_cross, 'Visible', 'on');drawnow;
    cue_sequence = [];
%    cue_sequence = createSequence(N/length(opt.speakerSelected));
    cue_sequence = stimutil_constraintSequence(length(opt.speakerSelected), N, 'margin',2);
    
    % play the target introduction audiostream
    fprintf('%3d. Target direction: %i\n', zz, target_sequence(zz));
    if isfield(speech, 'targetDirection'),
        PsychPortAudio('FillBuffer', opt.pahandle, speech.targetDirection);
        PsychPortAudio('Start', opt.pahandle);
        PsychPortAudio('Stop', opt.pahandle, 1);
    end
    
    pause(1);    
    if opt.visual_targetPresentation
       set(opt.handle_loc, 'FaceColor',opt.visual_colorInactive, 'Visible', 'on');
       set(opt.handle_loc(target_sequence(zz)), 'FaceColor', opt.visual_colorActive);
       drawnow;
    end
    
    % play the cuesound from the target direction
    waitForSync;
    for repI = 1:opt.repeatTarget,
      PsychPortAudio('FillBuffer', opt.pahandle, cueList(target_sequence(zz)).wav);
      PsychPortAudio('Start', opt.pahandle);
      ppTrigger(90+repI);
      PsychPortAudio('Stop', opt.pahandle, 1);
      waitForSync(opt.isi+opt.isi_jitter/2);
    end

    if opt.visual_targetPresentation
       set(opt.handle_loc(target_sequence(zz)), 'FaceColor', opt.visual_colorInactive);
       if ~opt.visual_cuePresentation,
         set(opt.handle_loc, 'Visible', 'off');
       end
       drawnow;
    end
    waitForSync(opt.delayAfterTargetPresentation);
    
    if opt.req_response, 
      [dmy] = acquire_bv(state); %clear the que
    end
    
    % start the actual experiment loop
    trial_duration= 100;  %% dummy value for first stimulus presentation
    for i= 1:N,
      PsychPortAudio('FillBuffer', opt.pahandle, cueList(cue_sequence(i)).wav);
      waitForSync(trial_duration);
      trial_duration = opt.isi + rand()*opt.isi_jitter;
      PsychPortAudio('Start', opt.pahandle);
      if cue_sequence(i) == target_sequence(zz),
        ppTrigger(cue_sequence(i)+20);
      else
        ppTrigger(cue_sequence(i));
      end
      if opt.visual_cuePresentation,
        set(opt.handle_loc(cue_sequence(i)), 'FaceColor', opt.visual_colorActive);
        drawnow;
      end
      if opt.req_response,
        startTime = clock;
        isdown = 0;
        resp = [];
        
        while isempty(resp) && 1000*etime(clock,startTime)<opt.resp_latency,
          [dmy,bn,mp,mt,md]= acquire_bv(state);
          for mm= 1:length(mt),
            resp= strmatch(mt{mm}, opt.response_markers);
            if ~isempty(resp),
              continue;
            end
          end
          pause(0.001);  %% this is to allow breaks by <Ctrl-C>
        end
        endTime = clock;
     
        if isempty(resp),
          if cue_sequence(i) ~= target_sequence(zz),
            correctResponse(zz, i) = 1;
            buttonPressed(zz, i) = 0;
            fprintf('Counted %i correct miss.\n', ...
                    length(find(abs(buttonPressed(zz,1:i)-1) .* ...
                                correctResponse(zz,1:i))));
          else
            if cue_sequence(i) == target_sequence(zz),
              correctResponse(zz, i) = 0;
              buttonPressed(zz, i) = 0;
              fprintf('Counted %i false negative.\n', ...
                      length(find(abs(buttonPressed(zz,1:i)-1) .* ...
                                  abs(correctResponse(zz,1:i)-1))));
            end
          end
        else
          if cue_sequence(i) == target_sequence(zz),
            correctResponse(zz, i) = 1;
            buttonPressed(zz, i) = 1;
            reactionTime(zz, i) = etime(endTime, startTime)*1000;
            fprintf('Counted %i correct hit.\n', ...
                    length(find(buttonPressed(zz,1:i) .* ...
                                correctResponse(zz,1:i))));
            fprintf('Reactiontime was: %.0f msec\n', reactionTime(zz,i));
          else
            correctResponse(zz, i) = 0;
            buttonPressed(zz, i) = 1;
            reactionTime(zz, i) = etime(endTime, startTime)*1000;
            fprintf('Counted %i false positive.\n', ...
                    length(find(buttonPressed(zz,1:i) .* ...
                                abs(correctResponse(zz,1:i)-1))));
            fprintf('Reactiontime was: %.0f msec\n', reactionTime(zz,i));
          end
        end
      end
      PsychPortAudio('Stop', opt.pahandle, 1);
      if opt.visual_cuePresentation,
        set(opt.handle_loc(cue_sequence(i)), 'FaceColor', opt.visual_colorInactive);
        drawnow;
      end
    end
    
    if opt.req_response,
      falseN = length(find(abs(buttonPressed(zz,:)-1) .* ...
                           abs(correctResponse(zz,:)-1)));
      falseP = length(find(buttonPressed(zz,:) .* ...
                           abs(correctResponse(zz,:)-1)));
      trueN = length(find(abs(buttonPressed(zz,:)-1) .* ...
                          correctResponse(zz,:)));
      trueP = length(find(buttonPressed(zz,:) .* ...
                          correctResponse(zz,:)));
      fprintf('True positives: %i\n', trueP);
      fprintf('True negatives: %i\n', trueN);
      fprintf('False positives: %i\n', falseP);
      fprintf('False negatives: %i\n', falseN);
    end
    
    pause(opt.duration_breakBetweenRounds/1000);
    outputCueSequence(zz,:) = cue_sequence;
end

ppTrigger(253);
if isfield(speech, 'trialStop'),
  PsychPortAudio('FillBuffer', opt.pahandle, speech.trialStop);
  PsychPortAudio('Start', opt.pahandle);
  PsychPortAudio('Stop', opt.pahandle, 1);
end
pause(0.1);

cue_sequence = outputCueSequence;
pause(1);
if ~opt.test & ~isempty(opt.filename),
    bvr_sendcommand('stoprecording');
end

pause(3);
% delete(h_msg);
