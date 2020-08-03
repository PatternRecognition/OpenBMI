%% Run setup
global TODAY_DIR REMOTE_RAW_DIR SESSION_TYPE DATA_DIR VP_CODE
acq_makeDataFolder('log_dir',1, 'multiple_folders', 1);
REMOTE_RAW_DIR= TODAY_DIR;

addpath([BCI_DIR '/acquisition/stimulation/spatial_auditory_p300/commons/']);
addpath([BCI_DIR '/acquisition/setups/SequenceComplexity/']);
opt.nrRuns = 3;

opt.speakerCount = 8;
opt.soundcard = 'ASIO';
opt.outputOffset = 0; % set automatically below
opt.mapping = [4 7 1 3 6 2 8 5];
% opt.mapping = [1:8];
opt.toneDuration = 40;


% High noise better audible
lowBase = 3000;
highBase = 7500;
toneStart = 440;
toneSteps = 3;
steps = 6; %percent change on boundary
stepsHigh = 7;

for speaker = 1:opt.speakerCount,
    noise = stimutil_filteredNoise(44100, opt.toneDuration/1000, 3, lowBase, highBase, 3, 3);
    tone = stimutil_generateTone(toneStart*(toneSteps^(1/12))^((speaker-1)*2), 'duration', opt.toneDuration, 'pan', [1], 'fs', 44100, 'rampon', 3, 'rampoff', 3);
    opt.sounds{find(opt.mapping == speaker)} = noise + tone';
    lowBase = lowBase+((steps/100)*lowBase);
    highBase = highBase+((stepsHigh/100)*highBase);    
end

opt.endTone = stimutil_generateTone(300, 'duration', 400, 'pan', [1], 'fs', 44100, 'rampon', 3, 'rampoff', 3)'*.3;
%opt.endTone = stimutil_generateTone(300, 'duration', 400, 'pan', [1], 'fs', 44100, 'rampon', 50, 'rampoff', 50)';

InitializePsychSound(1);
deviceList = PsychPortAudio('GetDevices');
multiChannIdx = find([deviceList.NrOutputChannels] >= opt.speakerCount);

if isempty(multiChannIdx)
    error('I''m sorry, no soundcard is detected that supports %i channels', opt.speakerCount);
end

if strmatch(deviceList(multiChannIdx).DeviceName, 'MOTU Audio ASIO'),
    opt.outputOffset = 2;
end

try
    dummy = PsychPortAudio('GetStatus', 0);
    PsychPortAudio('Close');
catch
end

ii = 1;
opt.pahandle = -1;
while (ii <= length({deviceList(multiChannIdx).DeviceName})) && (opt.pahandle < 0),
    if isequal(deviceList(multiChannIdx(ii)).HostAudioAPIName, opt.soundcard)
        % open soundcard
        opt.pahandle = PsychPortAudio('Open', multiChannIdx(ii)-1, 1, 2, 44100, [opt.speakerCount+opt.outputOffset 2], 0); 
        fprintf('Congratulations, the %s soundcard has successfully been detected and initialized.\n', deviceList(multiChannIdx(ii)).DeviceName);
        fprintf('This soundcard ensures low-latency, multi channel analog output and/or input.\n');
    end
    ii = ii + 1;
end

if opt.pahandle < 0,
    error('I''m sorry, no soundcard is detected that supports %i channels', opt.speakerCount);
end

%% Show experiment
rundef = merge_structs(opt,seq_generation());
stimutil_playSpatialSequence(opt.sounds, rundef, 'controlBV', 0);

%% Run experiment
for run = 1:opt.nrRuns,
    rundef = merge_structs(opt,seq_generation());
    save([TODAY_DIR '/rundefinition' num2str(run) '.mat'], 'rundef');
    stimutil_playSpatialSequence(opt.sounds, rundef, 'controlBV', 1);
end