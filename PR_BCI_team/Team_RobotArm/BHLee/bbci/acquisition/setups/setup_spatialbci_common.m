% set the bci dir (should exist)
startup_bbcilaptop; %% needed for acquire_bv

global TODAY_DIR REMOTE_RAW_DIR SESSION_TYPE DATA_DIR VP_CODE
% DATA_DIR = 'c:\test\';
% BCI_DIR = 'C:\Dokumenten\SVN\BBCI\';
SOUND_DIR = [BCI_DIR 'acquisition\data\sound\'];

% 
% bvr_sendcommand('loadworkspace', 'martijns_study_VPfa');
% try,
%   bvr_checkparport('type','S');
% catch
%   error(sprintf('BrainVision Recorder must be running.\nThen restart %s.', mfilename));
% end  


acq_getDataFolder('multiple_folders',1);
REMOTE_RAW_DIR= TODAY_DIR;

if ~exist('VP_CODE', 'var'),
    VP_CODE = input('Please enter a valid VP code: ', 's');
end
opt.subjectId = VP_CODE;
%opt.filename = [SESSION_TYPE];
opt.filename = 'OnlineTestFile';
opt.singleSpeaker = false;
opt.test = false;
opt.speakerCount = 8;
opt.speakerSelected = [7 8 1 2 3 4 5 6]; % first is lowest tone, last highest.
if opt.speakerCount < length(opt.speakerSelected),
    warning(sprintf('Can''t assign %i speakers. Only %i speakers are initialized.', length(opt.speakerSelected), opt.speakerCount));
end
opt.speakerName = {'Front', 'Front-right', 'Right', 'Back-right', 'Back', 'Back-left', 'Left', 'Front-left'};
opt.response_markers = {'R  1'};
opt.soundcard = 'ASIO'; %'M-Audio FW ASIO' or 'Realtek HD Audio output' or 'M-Audio FW ASIO' or 'ASIO4ALL v2'
if ~exist('audioIOmode', 'var'),
    audioIOmode = str2num(input('Give audio mode (1=playback, 2=record, 3=both): ', 's'));
end
opt.deviceMode = audioIOmode; % 1=playback, 2=record, 3=full duplex
opt.isi = 175; % inter stimulus interval
opt.isi_jitter = 0; % defines jitter in ISI
opt.fs = 44100;
opt.itType = 'fixed';
opt.mode = 'free';
opt.toneDuration = 40; % in ms
opt.countdown = 0;
opt.repeatTarget = 3;
opt.language = 'english'; % can be 'english' or 'german'
% opt.language = 'german'; % can be 'english' or 'german'
opt.speech_dir = [SOUND_DIR lower(opt.language) '/upSampled'];
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
%opt.position = [-1919 5 1920 1210];  % large monitor in TU lab
% opt.position = [-1279 30 1280 1019];  % small mobile monitor for external experiments
opt.position = [300 300 300 300]; %same monitor for testing
opt.background = [0 0 0];
opt.visualAid = false; 

% parameters for keyboard response, overrides ISI from above
opt.req_response = false;
opt.resp_latency = 2000;

%For experiment
%% multiple speakers, no tone overlay
% opt.cueStream = stimutil_filteredNoise(44100, opt.toneDuration/1000, 3, 150, 8000, 3, 3);
%% multiple speakers, tone overlay
lowBase = 320;
highBase = 2500;
toneStart = 440;
toneSteps = 2^(1/12);
steps = 30; %percent change on boundary
stepsHigh = 30;

for i = 1:length(opt.speakerSelected),
    speaker = opt.speakerSelected(i);
    opt.cueStream(speaker,:) = stimutil_filteredNoise(44100, opt.toneDuration/1000, 3, lowBase, highBase, 3, 3);
    tmpTone = stimutil_generateTone(toneStart*(2^(1/12))^((i-1)*2), 'duration', opt.toneDuration, 'pan', [1], 'fs', 44100, 'rampon', 3, 'rampoff', 3);
    toneOverlay = tmpTone(1:length(opt.cueStream(speaker,:)),1);
    opt.cueStream(speaker,:) = opt.cueStream(speaker,:) + (toneOverlay' * 0.15);
    lowBase = lowBase+((steps/100)*lowBase);
    highBase = highBase+((stepsHigh/100)*highBase);
end

clear lowBase highBase toneStart toneSteps steps stepsHigh;

%opt.speech.targetDirection = 'target_direction';
opt.speech.trialStart = 'trial_start';
opt.speech.trialStop = 'over';
opt.speech.relax = 'relax_now';
opt.speech.nextTrial = 'next_trial';
% % opt.speech.trialStart = 'start';
% % opt.speech.trialStop = 'vorbei';
% % opt.speech.relax = 'entspannen';
% % opt.speech.nextTrial = 'neustart';
%opt.speech.intro = 'trial_start';

SPEECH_DIR = [SOUND_DIR lower(opt.language) '/upSampled'];

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