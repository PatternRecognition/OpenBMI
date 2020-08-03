function history = auditory_MainRoutine(varargin)
%
% Synopsis: this is the main routine for running auditory P300 experiments.
% It is very flexible and can be used for multi directional as well as
% single direction (traditional) oddball paradigms. An application can be
% created as a 'plugin' to this routine, making it extensible. 
%
% use:
%    history = auditory_MainRoutine(opt)
%
% INPUT
%    OPT
%     .mode          Defines the operation mode. Can be 'copy' or
%                    'free' (=default)
%     .itType        Defines the how many iterations will be done. When 
%                    'fixed'(=default), the number of iterations is
%                    'maxRounds'. When 'continuous', an unending loop of
%                    rounds is entered, with a decision after each trial of
%                    block. When 'adaptive', a minimum of 'minRounds'
%                    rounds is done. Then, a new iteration is started only
%                    if the confidence of the decision has not passed
%                    'probThres' or the number if iterations > 'maxRounds'.
%     .application   Name of the application plugin to be used. A pluging
%                    consists of at least 3 files ('load_LUT',
%                    'post_processes' and 'pre_processes'). They reside in
%                    a subdirectory of the AEP dir whos name is equal to
%                    this variable. Files are loaded on run-time.
%     .spellString   For spelling applications in copy mode, a string can
%                    be given that should be copied. The necessary targets
%                    are found in the lookup table (LUT).
%     .targetSeq     In contrast to spellString, the targetSeq directly
%                    sets the steps that should be copied.
%     .contMemory    Only applicable in continuous mode. Sets the number of
%                    iterations prior to the current one that should be
%                    used. When 1 [=default], single-trial analyzis is
%                    done.
%     .minRounds     In adaptive mode, this sets the number of iterations
%                    that should minimally be performed before a decision
%                    could be made. [default=3]
%     .maxRounds     Number of iterations that should be performed
%                    (maximally, in adaptive mode) [default=10]
%     .probThres     In adaptive mode, the confidence in the decision
%                    should cross this threshold before a selection is
%                    made. [default = .5 ?]
%     .dTime         Sets the time of decision making in continuous mode.
%                    Can be after every single stimulus ('trialwise') or
%                    after every new iteration ('blockwise'=default).
%     .dataPort      Sets the UDP port where the classifier outcomes from
%                    BBCI_BET_APPLY come in [default=12345].
%     .testConnect   If true, send a marker at the very beginning and wait
%                    for a result on the dataPort. If none is received, the
%                    connection could not be correct and an error is thrown
%     .speakerSelected Set the numbers of the speakers that should be used.
%                    [default=[1:6]]
%     .randomClass   Boolean. If true, a random 'classifier' output is send
%                    to the dataPort upon every stimulus. This way, the 
%                    setup can be tested without having to run
%                    BBCI_BET_APPLY [default=0]
%
%   The opt variable is passed to the application specific pre_processes
%   and post_processes functions. Therefore, application specific variables
%   can be set in this struct.
%
%   Some special states and selections that can be set in post_processes
%      States
%         -1:       Terminate the routine
%
%      Selections
%         -1:       Delete the last selected item
%
% NOTE
% opt.pahandle can be obtained with the following functions (if the
% PsychToolbox is installed):
%
% InitializePsychSound(1);
% pahandle = PsychPortAudio('Open' [, deviceid][, mode][,reqlatencyclass][, freq][, channels][, buffersize][, suggestedLatency][, selectchannels]);
%
%
% Martijn Schreuder, 11/08/2009

global VP_CODE BCI_DIR DATA_DIR TODAY_DIR

opt= propertylist2struct(varargin{:});

opt= set_defaults(opt, ...
                  'mode', 'free', ...               % free, copy
                  'itType', 'fixed', ...            % fixed, adaptive, continuous
                  'spellString', '', ...            % copyspell the followin string
                  'application', 'HEXO_spell', ...  % set name of application
                  'contMemory', 1, ...              % in continuous mode, how many rounds should be remembered
                  'minRounds', 3, ...               % in adaptive mode, set a minimal amount of rounds
                  'maxRounds', 10, ...              % maximum number of rounds
                  'probThres', 0.5, ...             % the probability threshold for selection in adaptive mode
                  'dataPort', 12345, ...
                  'dTime', 'blockwise', ...         % Time of decision. 'Blockwise': after a block. 'Trialwise': after each trial
                  'procVar', struct(), ...
                  'targetSeq', [], ...
                  'testConnect', false, ...
                  'speakerSelected', [1:6], ...
                  'targetSeq', '', ...
                  'bv_host', ' 127.0.0.1', ...
                  'randomClass', 0);            

if strcmp(opt.itType, 'continuous') && strcmp(opt.mode, 'copy'),
    error('Continuous copy spelling not allowed');
elseif strcmp(opt.application, 'TRAIN') && (~strcmp(opt.itType, 'fixed') || ~strcmp(opt.mode, 'copy')),
    error('Training can only be in copy mode with fixed number of iterations.');
end
if ~isempty(opt.targetSeq) && ~isempty(opt.spellString),
    error('targetSeq and spellString can not be set at the same time.');
end
if strcmp(opt.application(1:5), 'TRAIN'),
    online = false;
else
    online = true;
end

%% Initialization 
% Load directories
opt.scriptroot = [BCI_DIR '\acquisition\stimulation\spatial_auditory_p300\'];
addpath([opt.scriptroot '\commons\']);
addpath([opt.scriptroot '\' opt.application '\']);
waitForSync;                                        % Initialize the timing clock
              
% Initialize the routine variables
run = true;
currentState = 1;
history.selections = [];
history.dictonary = '';
history.state = currentState;
history.written = '';
if strcmp(opt.itType, 'continuous'),
    clsOut = NaN*ones([opt.contMemory, length(opt.speakerSelected)]);
else
    clsOut = NaN*ones([opt.maxRounds, length(opt.speakerSelected)]);
    clsTemplate = clsOut;
end

%% establish a connection to the classifier and brainproduct sw
if ~isempty(opt.dataPort) && online,
    try
        get_data_udp();
        opt.portHandle = get_data_udp(opt.dataPort);
    catch
        opt.portHandle = get_data_udp(opt.dataPort);
    end
end
% if ~isempty(opt.bv_host),
%     bvr_checkparport;
% end

if opt.test,
    fprintf('Warning: test option set true: EEG is not recorded!\n');
else
    if ~isempty(opt.filename),
        bvr_startrecording([opt.filename VP_CODE]);
        pause(2);
        ppTrigger(251);
        pause(2);
    else
        warning('!*NOT* recording: opt.filename is empty');
    end
end

%% Recompile and/or load the lookup table & dictionary
Lut = load_LUT(opt, 'store', true, 'directory', TODAY_DIR);
if exist('load_dict') == 2,
    Dict = load_dict('store', true, 'directory', TODAY_DIR); % can be empty
else
    Dict = [];
end

%% Precreate sufficiantly large cue sequence
presetSeq = createSequence(40, length(opt.speakerSelected), 'repeatable', 1);
if strcmp(opt.mode, 'copy'),
    if ~isempty(opt.targetSeq),
        targetSequence = opt.targetSeq;
    elseif ~isempty(opt.spellString),
        targetSequence = findReverseSequence(opt.spellString, Lut);
    else
        targetSequence = createSequence(2, length(opt.speakerSelected));
    end
end

%% Send random data for testing if necesary
if opt.randomClass,
    try
        send_data_udp;
        send_data_udp(get_hostname, opt.dataPort);
    catch
        send_data_udp(get_hostname, opt.dataPort);
    end
end

if opt.testConnect && online,
    send_data_udp([1010101]);
    pause(.5);
    testDat = get_data_udp(opt.portHandle, 0.1, 0);
    if ~isempty(testDat) && testDat == 1010101,
        fprintf('Succesfully receiving data on port: %i\n', opt.dataPort);
    else
        warning('No data received on port: %i\n', opt.dataPort);
    end
end

%% Set flags and counters
counter = 1;
trial = 1;
clsCounter = 0;
lastDecision = 0;
newTrial = true;

%% Start the routine loop (one cue per iteration)
while run,
    cueNr = noZeroMod(counter, length(presetSeq));
    if opt.randomClass,
        bias =1;
        if strcmp(opt.mode, 'copy') && presetSeq(cueNr) == targetSequence(trial),
            bias = -1;
        end
        udpData = [rand(1,5)*2*bias presetSeq(cueNr)];
        send_data_udp(udpData);
    end

    %% do pre-rules only if a decision has been made or it is the very
    %% first stimulation
    opt.labels = [];
    if (exist('clOut', 'var') && isfield(clOut, 'class_label')) || newTrial,
        if strcmp(opt.mode, 'copy')
            opt.procVar = pre_processes(currentState, Lut, Dict, history, opt, 'targetDir', targetSequence(trial));
        else
            opt.procVar = pre_processes(currentState, Lut, Dict, history, opt);
        end
        clear clOut;
        newTrial = false;
    end
    
    %% Play a cue
    if strcmp(opt.mode, 'copy'),
        opt = stim_oddballAuditorySpatial(opt, 'targetDir', targetSequence(trial), 'cueDir', presetSeq(cueNr));
    else       
        opt = stim_oddballAuditorySpatial(opt, 'cueDir', presetSeq(cueNr));
    end
    
    %% Get a decision from the udp cue
    if online,
        tmpData = empty_udp_cue(opt.portHandle);
        if ~isempty(tmpData),
            switch opt.itType,
                case 'continuous',
                    row = noZeroMod(floor((clsCounter)/length(opt.speakerSelected)+1), opt.contMemory);
                otherwise,
                    row = noZeroMod(floor((clsCounter)/length(opt.speakerSelected)+1), opt.maxRounds);
            end
            clsOut(row, tmpData(6)) = tmpData(5);
            clsCounter = clsCounter+1;
        end
    end
    
    %% If a decision can/should be made, make it!
    % Continuous conditions    
    if strcmp(opt.itType, 'continuous'), 
         switch opt.dTime,
             case 'blockwise'
                 if noZeroMod(clsCounter, length(opt.speakerSelected)) == length(opt.speakerSelected),
                     [clOut.class_label, clOut.prob, clOut.vec] = util_selectClass(clsOut, 'mapping', opt.speakerSelected);
                 end
             case 'trialwise'
                 if clsCounter >= length(opt.speakerSelected),
                     [clOut.class_label, clOut.prob, clOut.vec] = util_selectClass(clsOut, 'mapping', opt.speakerSelected);
                 end
         end
    else
    %% fixed number of iterations per trial
        if strcmp(opt.itType, 'fixed') && ~mod(counter, opt.maxRounds*length(opt.speakerSelected)),
            if online,
                %% wait for the buffer to fill up and make decision
                while noZeroMod(clsCounter, opt.maxRounds*length(opt.speakerSelected)) < opt.maxRounds*length(opt.speakerSelected),
                    tmpData = empty_udp_cue(opt.portHandle);
                    if ~isempty(tmpData),                
                        row = noZeroMod(floor((clsCounter)/length(opt.speakerSelected)+1), opt.maxRounds);
                        clsOut(row, tmpData(6)) = tmpData(5);
                        clsCounter = clsCounter+1;
                        pause(0.05);
                    end             
                end
                trial = trial+1;
                [clOut.class_label, clOut.prob, clOut.vec] = util_selectClass(clsOut);
            else % offline session
                clOut.class_label = 1;clOut.prob = 1;clOut.vec = 1;
                trial = trial+1;
                newTrial = true;
            end
        elseif strcmp(opt.itType, 'adaptive') && (clsCounter-lastDecision >= opt.minRounds*length(opt.speakerSelected)),
        %% Adaptive number of iterations per trial      
            if ~mod(clsCounter-lastDecision, length(opt.speakerSelected)),
                [tmpClass, clOut.prob, tmpVec] = util_selectClass(clsOut, 'mapping', opt.speakerSelected);
                if (clOut.prob >= opt.probThres) || (clsCounter-lastDecision)==opt.maxRounds*length(opt.speakerSelected),
                    clOut.class_label = tmpClass; clOut.vec = tmpVec;
                    %clear UDP cue
                    while clsCounter < counter,
                        dmy = get_data_udp(opt.portHandle, .01, 0);
                        if ~isempty(dmy),
                            clsCounter = clsCounter+1;
                        end
                        pause(0.05);
                    end
                    trial = trial+1;
                    % set counter to next full blockstart
                    counter = counter+(length(opt.speakerSelected)-mod(counter, length(opt.speakerSelected)));
                    clsCounter = counter;
                    lastDecision = counter;
                    clsOut = clsTemplate;                    
                end
            end
        end %non-continuous
    end % decision
    
    %% do post-rules
    if (exist('clOut', 'var') && isfield(clOut, 'class_label')) || newTrial,
%         vecT = [vecT, vecT(:,end)+[clOut.vec']]; %% temp hack
        [newState, choice, opt.procVar] = post_processes(clOut, currentState, Lut, history, opt);
        history.selections = [history.selections, clOut.class_label];
        history.state = [history.state newState];
        if choice == -1,
            history.written = history.written(1:end-1);            
        else
            history.written = [history.written choice];
        end
        currentState = newState; 
    end
    
    %% write logfile
    
    if currentState == -1, 
        run = 0;
    elseif strcmp(opt.mode, 'copy') && trial > length(targetSequence)
        run = 0;
    end
    counter = counter + 1;
end


%% Commands before closing down
rmpath([opt.scriptroot '\' opt.application '\']);
opt = rmfield(opt, 'procVar');
if online,
    get_data_udp(opt.portHandle);
    send_data_udp;
end
if ~opt.test && ~isempty(opt.filename),
    bvr_sendcommand('stoprecording');
    ppTrigger(254);
end

 end

 
%% Set helperfunctions
function idx = noZeroMod(Nr, dev)
    if Nr ~= 0,
        idx = mod(Nr, dev);
        if idx == 0,
            idx = dev;
        end
    else
        idx = Nr;
    end
end

function udp = empty_udp_cue(varargin),
    portHandle = varargin{1};
    if nargin == 2,
        timeOut = varargin{2};
    end
    udp = NaN*ones([1,6]);
    tic;
    while ~isempty(udp) && (udp(6) == 0 || isnan(udp(6))),
        if exist('timeOut', 'var') && toc/1000 > timeOut,
            break;
        end
        udp = get_data_udp(portHandle, 0.001, 0);
    end
end
        


