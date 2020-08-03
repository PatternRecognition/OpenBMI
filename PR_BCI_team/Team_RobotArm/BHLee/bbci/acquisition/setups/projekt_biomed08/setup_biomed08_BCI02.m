addpath([BCI_DIR 'acquisition\stimulation\projekt_biomed08\']);

%%--
if 0,
bvr_sendcommand('loadworkspace', 'projekt_biomed08');
try,
  bvr_checkparport('type','S');
catch
  error(sprintf('BrainVision Recorder must be running.\nThen restart %s.', mfilename));
end  
SOUND_DIR = [BCI_DIR 'acquisition\data\sound\'];


global TODAY_DIR
acq_makeDataFolder;
end

%% Set general options for the run of the experiment
N = 75;
trials = 5;


opt.subjectId = VP_CODE;
opt.rampoff= 20;
opt.cuelength= 250;
opt.singleSpeaker = 0;
opt.test = false;
opt.speakerCount = 6;
opt.speakerSelected = [1:6]; % first is lowest tone, last highest.
if opt.speakerCount < length(opt.speakerSelected),
    warning(sprintf('Can''t assign %i speakers. Only %i speakers are initialized.', length(opt.speakerSelected), opt.speakerCount));
end
opt.speakerName = cprintf(1:opt.speakerCount);
opt.response_markers = {'R  1'};
opt.soundcard = 'M-Audio FW ASIO'; %'M-Audio FW ASIO' or 'Realtek HD Audio output' or 'M-Audio FW ASIO' or 'ASIO4ALL v2'
opt.isi = 250; % inter stimulus interval
opt.isi_jitter = 0; % defines jitter in ISI
opt.fs = 44100;
opt.toneDuration = 40; % in ms
%opt.dualStim = false; %
%opt.dualDistance = 1;
opt.countdown = 5;
opt.repeatTarget = 3;
%opt.language = 'english'; % can be 'english' or 'german'
opt.language = 'german'; % can be 'english' or 'german'
opt.speech_dir = [SOUND_DIR lower(opt.language) '/upSampled'];
opt.calibrated = diag(ones(1, opt.speakerCount));
opt.writeCalibrate = false;

opt.position = VP_SCREEN;
opt.background = [0 0 0];
opt.visual_cuePresentation = true; 

% parameters for keyboard response, overrides ISI from above
opt.req_response = false;
opt.resp_latency = 2000;

[cuewav, cuefs]= wavread([SOUND_DIR '/cues_tactile/250Hz_1000ms.wav']);
if cuefs~=opt.fs,
  error('mismatch in sampling rate');
end
Nwav= ceil(opt.cuelength/1000*opt.fs);
if size(cuewav,1)<Nwav,
  error('cue wave too short');
end
cuewav= cuewav(1:Nwav,1)';
Noff= round(opt.rampoff/1000*opt.fs);
ramp= cos((1:Noff)*pi/Noff/2).^2;
cuewav(end-Noff+1:end)= cuewav(end-Noff+1:end) .* ramp;
opt.cueStream= repmat(cuewav, [length(opt.speakerSelected) 1]);

%For experiment
%% multiple speakers, no tone overlay
% opt.cueStream = stimutil_filteredNoise(44100, opt.toneDuration/1000, 3, 150, 8000, 3, 3);
%% multiple speakers, tone overlay
%lowBase = 320;
%highBase = 2500;
%toneStart = 250;
%steps = 0; %percent change on boundary
%stepsHigh = 0;

%for i = 1:length(opt.speakerSelected),
%  opt.cueStream(i,:)= cuewav;
%    opt.cueStream(i,:) = stimutil_filteredNoise(44100, opt.toneDuration/1000, 3, lowBase, highBase, 3, 3);
%    tmpTone = stimutil_generateTone(toneStart*(2^(1/12))^((i-1)*2), 'duration', opt.toneDuration, 'pan', [1], 'fs', 44100, 'rampon', 3, 'rampoff', 3);
%    toneOverlay = tmpTone(1:length(opt.cueStream(i,:)),1);
%    opt.cueStream(i,:) = opt.cueStream(i,:) + (toneOverlay' * 0.15);
%    lowBase = lowBase+((steps/100)*lowBase);
%    highBase = highBase+((stepsHigh/100)*highBase);
%end

%opt.speech.targetDirection = 'target_direction';
%opt.speech.trialStart = 'trial_start';
%opt.speech.trialStop = 'over';
%opt.speech.relax = 'relax_now';
%opt.speech.nextTrial = 'next_trial';
%opt.speech.trialStart = 'start';
%opt.speech.trialStop = 'vorbei';
%opt.speech.relax = 'entspannen';
%opt.speech.nextTrial = 'neustart';
%opt.speech.intro = 'trial_start';
opt.speech= [];

SPEECH_DIR = [SOUND_DIR lower(opt.language) '/upSampled'];

%% Find proper soundcard, for now only 'M-Audio FW ASIO'
% But could easily be extended to other multi channel devices
InitializePsychSound(1);
deviceList = PsychPortAudio('GetDevices');
multiChannIdx = find([deviceList.NrOutputChannels] >= opt.speakerCount);

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
    if isequal(deviceList(multiChannIdx(ii)).DeviceName, opt.soundcard)
        % open soundcard
        opt.pahandle = PsychPortAudio('Open', multiChannIdx(ii)-1, [], 2, 44100, opt.speakerCount, 0); 
        fprintf('Congratulations, the %s soundcard has successfully been detected and initialized.\n', deviceList(multiChannIdx(ii)).DeviceName);
        fprintf('This soundcard ensures low-latency, multi channel analog output\n');
    end
    ii = ii + 1;
end

if opt.pahandle < 0,
    error('I''m sorry, no soundcard is detected that supports %i channels', opt.speakerCount);
end

clear multiChannIdx deviceList oldDir dummy ii;
