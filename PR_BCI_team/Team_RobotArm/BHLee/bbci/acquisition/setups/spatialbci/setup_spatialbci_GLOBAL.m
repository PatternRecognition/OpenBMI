startup_bbcilaptop; %% needed for acquire_bv

global TODAY_DIR REMOTE_RAW_DIR SESSION_TYPE DATA_DIR VP_CODE
SOUND_DIR = [BCI_DIR 'acquisition\data\sound\'];

acq_getDataFolder('multiple_folders',1);
REMOTE_RAW_DIR= TODAY_DIR;

if ~exist('VP_CODE', 'var'),
    VP_CODE = input('Please enter a valid VP code: ', 's');
end
opt.subjectId = VP_CODE;
opt.test = false;

opt.speakerCount = 8;
% opt.speakerSelected = [4:6 1:3]; % first is lowest tone, last highest.
if ~isfield(opt, 'speakerSelected'),
  opt.speakerSelected = [6 2 4 1 5 3]; % first is lowest tone, last highest.
end
if opt.speakerCount < length(opt.speakerSelected),
    warning(sprintf('Can''t assign %i speakers. Only %i speakers are initialized.', length(opt.speakerSelected), opt.speakerCount));
end

opt.soundcard = 'ASIO'; %'M-Audio FW ASIO' or 'Realtek HD Audio output' or 'M-Audio FW ASIO' or 'ASIO4ALL v2'
if ~exist('audioIOmode', 'var'),
    audioIOmode = str2num(input('Give audio mode (1=playback, 2=record, 3=both): ', 's'));
end
opt.deviceMode = audioIOmode; % 1=playback, 2=record, 3=full duplex
opt.fs = 44100;

if ~isfield(opt, 'toneDuration'),
  opt.toneDuration = 20; % in ms
end

% opt.language = 'german'; % can be 'english' or 'german'
if ~isfield(opt, 'language'),
  opt.language = 'english'; % can be 'english' or 'german'
end
opt.speech_dir = [SOUND_DIR lower(opt.language) '/upSampled'];
SPEECH_DIR = opt.speech_dir;

try
    fileName = [BCI_DIR 'acquisition/stimulation/spatial_auditory_p300/CalibrationFiles/' VP_CODE 'Calibration.dat'];
    opt.calibrated = diag(load([fileName]));
catch
    opt.calibrated = diag(load([BCI_DIR 'acquisition/stimulation/spatial_auditory_p300/CalibrationFiles/calibratedParam.dat'])');
    warning('No subject specific speakercalibration found. Using standard calibration.\nType speaker_calibration(opt) for individualized calibration.');   
end
opt.writeCalibrate = true;
clear filename;

% set prefered voice
if strcmp(opt.language, 'german'),
    opt.voice = 'hmm-bits2';
    opt.languageIndicator = 'TEXT_DE';
else
    opt.voice = 'hmm-jmk';
    opt.languageIndicator = 'TEXT_EN';
end

% some onscreen parameters
% opt.position = [-1919 -110 1920 1300];  % large monitor in TU lab
% opt.position = [-1279 30 1280 1019];  % small mobile monitor for external experiments
% opt.position = [300 300 300 300]; %same monitor for testing
opt.background = [0 0 0];
opt.visualAid = false; 

%% multiple speakers, tone overlay
% High noise rather hidden
% lowBase = 2500;
% highBase = 5500;
% toneStart = 440;
% toneSteps = 3;
% steps = 8; %percent change on boundary
% stepsHigh = 12;

% High noise better audible
lowBase = 3000;
highBase = 7500;
toneStart = 440;
toneSteps = 3;
steps = 6; %percent change on boundary
stepsHigh = 7;

for i = 1:length(opt.speakerSelected),
    speaker = opt.speakerSelected(i);
    opt.cueStream(speaker,:) = stimutil_filteredNoise(44100, opt.toneDuration/1000, 3, lowBase, highBase, 3, 3);
%     opt.cueStream(speaker,:) = stimutil_filteredNoise(44100, opt.toneDuration/1000, 3, 1900, 12000, 3, 3);
    tmpTone = stimutil_generateTone(toneStart*(toneSteps^(1/12))^((i-1)*2), 'duration', opt.toneDuration, 'pan', [1], 'fs', 44100, 'rampon', 3, 'rampoff', 3);
    toneOverlay = tmpTone(1:length(opt.cueStream(speaker,:)),1);
    opt.cueStream(speaker,:) = opt.cueStream(speaker,:) + (toneOverlay' * 0.15);
%     opt.cueStream(speaker,:) = opt.cueStream(speaker,:);
    lowBase = lowBase+((steps/100)*lowBase);
    highBase = highBase+((stepsHigh/100)*highBase);
end

clear lowBase highBase toneStart toneSteps steps stepsHigh;

opt.speech.trialStart = 'trial_start';
opt.speech.trialStop = 'over';
opt.speech.relax = 'relax_now';
opt.speech.nextTrial = 'next_trial';

%% Find proper soundcard, for now only 'M-Audio FW ASIO'
% But could easily be extended to other multi channel devices
InitializePsychSound(1);
deviceList = PsychPortAudio('GetDevices');
multiChannIdx = find([deviceList.NrOutputChannels] >= opt.speakerCount);
recChannIdx = find([deviceList.NrInputChannels] >= 2);

if isempty(multiChannIdx)
    error('I''m sorry, no soundcard is detected that supports %i channels', opt.speakerCount);
end

%% Check if a soundcard has already be initialized
% If so, close it and use the 'M-Audio FW ASIO'
try
    dummy = PsychPortAudio('GetStatus', 0);
    PsychPortAudio('Close');
catch
end

%% Use first soundcard that has the set name and number of channels
ii = 1;
opt.pahandle = -1;
while (ii <= length({deviceList(multiChannIdx).DeviceName})) && (opt.pahandle < 0),
    if isequal(deviceList(multiChannIdx(ii)).HostAudioAPIName, opt.soundcard)
        % open soundcard
        opt.pahandle = PsychPortAudio('Open', multiChannIdx(ii)-1, opt.deviceMode, 2, 44100, [opt.speakerCount 2], 0); 
        fprintf('Congratulations, the %s soundcard has successfully been detected and initialized.\n', deviceList(multiChannIdx(ii)).DeviceName);
        fprintf('This soundcard ensures low-latency, multi channel analog output and/or input.\n');
    end
    ii = ii + 1;
end

if opt.pahandle < 0,
    error('I''m sorry, no soundcard is detected that supports %i channels', opt.speakerCount);
end

clear multiChannIdx deviceList oldDir dummy ii i recChannIdx tmpTone toneOverlay;