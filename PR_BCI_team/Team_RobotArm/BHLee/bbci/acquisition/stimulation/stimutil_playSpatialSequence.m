function stimutil_playSpatialSequence(sounds, varargin),

% Synopsis: uses the PsychToolbox PsychPortAudio driver to play a sound
% sequence (or multiple sounds in a cell array) through the indicated 
% speakers.
% With the M-Audio FireWire 410 up to 8 speakers can be individually
% used, or any combination of those.
%
% use:
%    stimutil_playSpatialSequence(sounds, opt);
%
% INPUT
%    sounds          Cell array with the different sounds that should
%                    be played
%    OPT
%     .SOA           Defines the stimulus-onset asynchrony (ie. the time
%                    between stimulus onsets).
%     .sequence      Defines the sequence that is presented. If it is a
%                    cell, each element is assumed to be the sequence of a
%                    new trial.
%     .target        Sets the target direction. Can be a scalar (in which
%                    case the target will be equal for all sequences), or a
%                    vector with the same length as .sequence.
%     .require_response If set to true, the experimenter will be promted to
%                    enter some value (numeric), that will be stored in the
%                    the EEG data.
%     .preplaySeq    If set to true, the sequence will be played before the
%                    start. Can be scalar, or a vector with length as .sequence.
%                    Requires seqsize to be set.
%     .preplaySOA    Default 500 [ms]. The pause in
%     .seqsize       Indicates the length of a single iteration. Relevant
%                    for preplaying the sequence. Must be a scalar or the
%                    same length as .sequence
%     .controlBV     If set to true, this function takes care of the
%                    starting and stopping of the recorder.
%     .filenames     If controlBV is true, this parameter sets the filename
%                    that is used. Again, it can be a single string or a
%                    cell array with the same size as .sequence.
%     .endTone       Vector containing the wave data for the tone
%                    indicating end of trial
%     .targetOffset  This offset is added to all target stimuli
%     .nontargetOffset This offset is added to all non-target stimuli
%     .responseOffset This offset is added to the response given by the
%                    subject
%     .targetPresents The amount of times that the target is presented
%                    before the trial
%     .betweenTargetPause The time between the target presentations
%     .beforeTrialPause The time between target presentation and trail
%                    start
%     .afterTrialPause The time between the end of a trial and the end tone
%     .pahandle      The handle to the soundcard opened with
%                    PsychPortAudio
%     .speakerCount  The number of speakers that have been loaded for
%                    the soundcard
%     .outputOffset  Some soundcards send the first X channels to special
%                    outputs, such as mains. This offset can counteract
%                    this.
%
% NOTE
% opt.pahandle can be obtained with the following functions (if the
% PsychToolbox is installed):
%
% InitializePsychSound(1);
% pahandle = PsychPortAudio('Open' [, deviceid][, mode][,reqlatencyclass][, freq][, channels][, buffersize][, suggestedLatency][, selectchannels]);
%
%
% Martijn Schreuder, 7/6/2012
 

    opt= propertylist2struct(varargin{:});
    opt= set_defaults(opt, ...
        'SOA', 200, ...
        'sequence', repmat([1:8],1,10), ...
        'target', 1, ...
        'requireResponse', 1, ...
        'controlBV', 1, ...
        'filenames', 'test', ...
        'endTone', [], ...
        'targetOffset', 10, ...
        'nontargetOffset', 0, ...
        'responseOffset', 100, ...
        'targetPresents', 3, ...
        'preplaySeq', 0, ...
        'seqsize', 8, ...
        'preplaySOA', 500, ...
        'betweenTargetPause', 500, ...
        'beforeTrialPause', 1500, ...
        'afterTrialPause', 1500, ...
        'speakerCount', 8, ...
        'pahandle', [], ...
        'outputOffset', 0);
    
    % Lots of sanity checks, just to be sure
    if ~iscell(sounds),
        sounds = {sounds};
    end
    if ~iscell(opt.filenames),
        opt.filenames = {opt.filenames};
    end
    if ~iscell(opt.sequence),
        opt.sequence = {opt.sequence};
    end
    
    if length(sounds) == 1,
        sounds = repmat(sounds, 1, opt.speakerCount);
    elseif length(sounds) > 1 && length(sounds) < opt.speakerCount,
        error('Sorry, I''m not sure where to place those sounds');
    end
    if length(opt.sequence) > 1,
        if length(opt.SOA) == 1,
            opt.SOA = repmat(opt.SOA, 1, length(opt.sequence));
        elseif length(opt.SOA) ~= length(opt.sequence),
            error('Sorry, I''m not sure how to match the SOA to the sequences');
        end
        if length(opt.target) == 1,
            opt.target = repmat(opt.target, 1, length(opt.sequence));
        elseif length(opt.target) ~= length(opt.sequence),
            error('Sorry, I''m not sure how to match the targets to the sequences');
        end
        if length(opt.preplaySeq) == 1,
            opt.preplaySeq = repmat(opt.preplaySeq, 1, length(opt.sequence));
        elseif length(opt.preplaySeq) ~= length(opt.sequence),
            error('Sorry, I''m not sure how to match the preplayseq to the sequences');
        end
        if length(opt.seqsize) == 1,
            opt.seqsize = repmat(opt.seqsize, 1, length(opt.sequence));
        elseif length(opt.seqsize) ~= length(opt.sequence),
            error('Sorry, I''m not sure how to match the preplayseq to the sequences');
        end        
        if opt.controlBV && length(opt.filenames) == 1,
            opt.filenames = repmat(opt.filenames, 1, length(opt.sequence));
        elseif opt.controlBV && length(opt.filenames) ~= length(opt.sequence),
            error('Sorry, I''m not sure how to match the filenames to the sequences');
        end        
    end
    
    % prepare the sounds
    stimuli = {};
    for i = 1:opt.speakerCount,
        stimuli{i} = zeros(opt.speakerCount+opt.outputOffset,length(sounds{i}));
        stimuli{i}(i+opt.outputOffset,:) = sounds{i};
    end
    if ~isempty(opt.endTone) && size(opt.endTone,1) == 1,
        opt.endTone = repmat(opt.endTone,opt.speakerCount+opt.outputOffset, 1);
    end
    
    for tr = 1:length(opt.sequence),
        % initialize the timer
        waitForSync;
        
        % start BV recording
        if opt.controlBV,
            realname = bvr_startrecording(opt.filenames{tr}, 'impedances', 0, 'append_VP_CODE', 1);
            pause(2);
        end
        
        % pause
        waitForSync(opt.beforeTrialPause);
        
        % play targets
        for trprs = 1:opt.targetPresents,
             PsychPortAudio('FillBuffer', opt.pahandle, stimuli{opt.target(tr)});
             waitForSync(opt.betweenTargetPause);
             PsychPortAudio('Start', opt.pahandle);
             PsychPortAudio('Stop', opt.pahandle, 1);                
        end
        
        % preplay sequence
        if opt.preplaySeq(tr),
            % pause
            waitForSync(opt.beforeTrialPause);            
            for prply = 1:opt.seqsize(tr),
                 PsychPortAudio('FillBuffer', opt.pahandle, stimuli{opt.sequence{tr}(prply)});
                 waitForSync(opt.preplaySOA);
                 PsychPortAudio('Start', opt.pahandle);
                 PsychPortAudio('Stop', opt.pahandle, 1);
            end
        end
        
        % pause
        waitForSync(opt.beforeTrialPause);
        
        % start sequence
        for sq = 1:length(opt.sequence{tr}),
            PsychPortAudio('FillBuffer', opt.pahandle, stimuli{opt.sequence{tr}(sq)});
            istrg = opt.sequence{tr}(sq) == opt.target(tr);
            waitForSync(opt.SOA(tr));
            ppTrigger(opt.sequence{tr}(sq) + ...
                (istrg*opt.targetOffset) + ...
                (~istrg*opt.nontargetOffset));
            PsychPortAudio('Start', opt.pahandle);
            PsychPortAudio('Stop', opt.pahandle, 1);            
        end
        
        % stop tone
        waitForSync(opt.afterTrialPause);        
        if ~isempty(opt.endTone),
            PsychPortAudio('FillBuffer', opt.pahandle, opt.endTone);
            PsychPortAudio('Start', opt.pahandle);
            PsychPortAudio('Stop', opt.pahandle, 1); 
        end
        
        % give input on quality
        if opt.requireResponse,
            rating = [];
            while isempty(rating),
                rating = str2num(input('What was the subjective rating [0 - 10]? ', 's'));
                if ~ismember(rating,[0:10]),
                    rating = [];
                    disp('Input must be between 0 and 10');
                end
            end
            ppTrigger(rating + opt.responseOffset);
        end
        
        % stop recording
        if opt.controlBV,
            pause(1);
            bvr_sendcommand('stoprecording');
        end            
        
        % promt for next trial
        input('Press ENTER to continue to the next trial. ');
    end
end