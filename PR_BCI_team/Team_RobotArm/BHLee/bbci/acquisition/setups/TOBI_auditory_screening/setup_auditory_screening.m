startup_bbcilaptop06b; %% needed for acquire_bv

global TODAY_DIR REMOTE_RAW_DIR SESSION_TYPE DATA_DIR VP_CODE
SOUND_DIR = [BCI_DIR 'acquisition\data\sound\'];

if ~exist('VP_CODE', 'var'),
    VP_CODE = input('Please enter a valid VP code: ', 's');
end


acq_getDataFolder('multiple_folders',1);
REMOTE_RAW_DIR= TODAY_DIR;


% Stimulation paradigm settings
opt.alternative_placing = 1;
opt.background = [0 0 0];
opt.avoid_dev_repetitions = 0;
opt.require_response = 0;
opt.fixation = 1;
opt.speech_intro = '';
opt.msg_fin = 'Ende'
opt.msg_intro = 'Entspannen';
opt.speech_dir = [BCI_DIR 'acquisition\data\sound\german\upSampled'];
opt.fs = 44100;
opt.speech.trialStart = 'trial_start';
opt.speech.trialStop = 'over';
opt.speech.relax = 'relax_now';
opt.speech.nextTrial = 'next_trial';
opt.bv_host = 'localhost';

% Soundcard settings
opt.soundcard = 'ASIO';
opt.deviceMode = 1; % playback
opt.fs = 44100;
opt.speaker_number = 2;

InitializePsychSound(1);
deviceList = PsychPortAudio('GetDevices');
multiChannIdx = find([deviceList.NrOutputChannels] >= 2);

% Check if a soundcard has already be initialized
% If so, close it and use the 'M-Audio FW ASIO'
try
    dummy = PsychPortAudio('GetStatus', 0);
    PsychPortAudio('Close');
catch
end

% Use first soundcard that has the set name and number of channels
ii = 1;
opt.pahandle = -1;
while (ii <= length({deviceList(multiChannIdx).DeviceName})) && (opt.pahandle < 0),
    if isequal(deviceList(multiChannIdx(ii)).HostAudioAPIName, opt.soundcard)
        % open soundcard
        opt.pahandle = PsychPortAudio('Open', multiChannIdx(ii)-1, opt.deviceMode, 2, 44100, [2 2], 0); 
        fprintf('Congratulations, the %s soundcard has successfully been detected and initialized.\n', deviceList(multiChannIdx(ii)).DeviceName);
        fprintf('This soundcard ensures low-latency, multi channel analog output and/or input.\n');
    end
    ii = ii + 1;
end

if opt.pahandle < 0,
    error('I''m sorry, no soundcard is detected that supports %i channels', 2);
end
