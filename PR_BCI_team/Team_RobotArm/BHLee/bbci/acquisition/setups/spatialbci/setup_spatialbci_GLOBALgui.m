startup_bbcilaptop; %% needed for acquire_bv

global TODAY_DIR REMOTE_RAW_DIR SESSION_TYPE DATA_DIR VP_CODE
SOUND_DIR = [BCI_DIR 'acquisition\data\sound\'];

acq_getDataFolder('multiple_folders',1);
REMOTE_RAW_DIR= TODAY_DIR;

% if ~exist('VP_CODE', 'var'),
%     VP_CODE = input('Please enter a valid VP code: ', 's');
% end
glo_opt.subjectId = VP_CODE;

glo_opt.speakerCount = 6;

if ~isfield(glo_opt, 'speakerSelected'),
  glo_opt.speakerSelected = [6 2 4 1 5 3]; % first is lowest tone, last highest.
end
if glo_opt.speakerCount < length(glo_opt.speakerSelected),
    warning(sprintf('Can''t assign %i speakers. Only %i speakers are initialized.', length(glo_opt.speakerSelected), glo_opt.speakerCount));
end

glo_opt.soundcard = 'ASIO'; %'M-Audio FW ASIO' or 'Realtek HD Audio output' or 'M-Audio FW ASIO' or 'ASIO4ALL v2'
audioIOmode = 1;
if ~exist('audioIOmode', 'var'),
    audioIOmode = str2num(input('Give audio mode (1=playback, 2=record, 3=both): ', 's'));
end
glo_opt.deviceMode = audioIOmode; % 1=playback, 2=record, 3=full duplex
glo_opt.fs = 44100;

glo_opt.speech_dir = [SOUND_DIR lower(glo_opt.language) '/upSampled'];
SPEECH_DIR = glo_opt.speech_dir;

try
    fileName = [TODAY_DIR 'Calibration.dat'];
    glo_opt.calibrated = diag(load([fileName]));
catch
    glo_opt.calibrated = diag(load([BCI_DIR 'acquisition/stimulation/spatial_auditory_p300/CalibrationFiles/calibratedParam.dat'])');
    warning('No subject specific speakercalibration found. Using standard calibration.\nType speaker_calibration(glo_opt) for individualized calibration.');   
end
glo_opt.writeCalibrate = true;
clear filename;

% set prefered voice
if strcmp(glo_opt.language, 'german'),
    glo_opt.voice = 'hmm-bits2';
    glo_opt.languageIndicator = 'TEXT_DE';
else
    glo_opt.voice = 'hmm-jmk';
    glo_opt.languageIndicator = 'TEXT_EN';
end

% some onscreen parameters
% glo_opt.position = [-1919 -110 1920 1300];  % large monitor in TU lab
% glo_opt.position = [-1279 30 1280 1019];  % small mobile monitor for external experiments
% glo_opt.position = [300 300 300 300]; %same monitor for testing
glo_opt.background = [0 0 0];
glo_opt.visualAid = false; 

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

for i = 1:length(glo_opt.speakerSelected),
    speaker = glo_opt.speakerSelected(i);
    glo_opt.cueStream(speaker,:) = stimutil_filteredNoise(44100, glo_opt.toneDuration/1000, 3, lowBase, highBase, 3, 3);
    tmpTone = stimutil_generateTone(toneStart*(toneSteps^(1/12))^((i-1)*2), 'duration', glo_opt.toneDuration, 'pan', [1], 'fs', 44100, 'rampon', 5, 'rampoff', 5);
    toneOverlay = tmpTone(1:length(glo_opt.cueStream(speaker,:)),1);
    glo_opt.cueStream(speaker,:) = glo_opt.cueStream(speaker,:) + (toneOverlay' * 0.15);
    lowBase = lowBase+((steps/100)*lowBase);
    highBase = highBase+((stepsHigh/100)*highBase);
end

clear lowBase highBase toneStart toneSteps steps stepsHigh;

glo_opt.speech.trialStart = 'trial_start';
glo_opt.speech.trialStop = 'over';
glo_opt.speech.relax = 'relax_now';
glo_opt.speech.nextTrial = 'next_trial';

%% Find proper soundcard, for now only 'M-Audio FW ASIO'
% But could easily be extended to other multi channel devices
InitializePsychSound(1);
deviceList = PsychPortAudio('GetDevices');
multiChannIdx = find([deviceList.NrOutputChannels] >= glo_opt.speakerCount);
recChannIdx = find([deviceList.NrInputChannels] >= 2);

if isempty(multiChannIdx)
    error('I''m sorry, no soundcard is detected that supports %i channels', glo_opt.speakerCount);
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
glo_opt.pahandle = -1;
while (ii <= length({deviceList(multiChannIdx).DeviceName})) && (glo_opt.pahandle < 0),
    if isequal(deviceList(multiChannIdx(ii)).HostAudioAPIName, glo_opt.soundcard)
        % open soundcard
        glo_opt.pahandle = PsychPortAudio('Open', multiChannIdx(ii)-1, glo_opt.deviceMode, 2, 44100, [glo_opt.speakerCount 2], 0); 
        fprintf('Congratulations, the %s soundcard has successfully been detected and initialized.\n', deviceList(multiChannIdx(ii)).DeviceName);
        fprintf('This soundcard ensures low-latency, multi channel analog output and/or input.\n');
    end
    ii = ii + 1;
end

if glo_opt.pahandle < 0,
    error('I''m sorry, no soundcard is detected that supports %i channels', glo_opt.speakerCount);
end

clear multiChannIdx deviceList oldDir dummy ii i recChannIdx tmpTone toneOverlay;