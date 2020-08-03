% Auditory ISI-experiment run-script
% run_TOBI_students_isi
%
% MAKE SURE YOU HAVE EXECUTED  THE CORRECT SETUP FILE ALREADY!
% setup_TOBI_students_isi

opt = [];

%% setup ASIO

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


%% Initialization
%TODO everything... call play_auditory_oddball_ISI with the specific
%parameters

NRUNS = 5;

opt.toneDuration = 40;
opt.speakerSelected = [6 2 4 1 5 3];
opt.language = 'german';

%setup_spatialbci_GLOBAL

opt.isi_jitter = 0; % defines jitter in ISI

opt.itType = 'fixed';
opt.mode = 'copy';
opt.application = 'TRAIN';

opt.countdown = 0;
% opt.repeatTarget = 3;

opt.require_response = 0;

opt.fixation = 1;
opt.filename = 'auditory_isi';
opt.speech_intro = '';
opt.fixation = 1;
opt.fs = 44100;
opt.cue_std = stimutil_generateTone(500, 'harmonics', 7, 'duration', 50, 'pan', 1, 'fs', opt.fs);
opt.cue_std = [opt.cue_std opt.cue_std];
opt.cue_std = opt.cue_std*.25;
opt.cue_dev = stimutil_generateTone(1000, 'harmonics', 7, 'duration', 50, 'pan', 1, 'fs', opt.fs);
opt.cue_dev = [opt.cue_dev opt.cue_dev];
opt.cue_dev = opt.cue_dev*.25;
opt.use_speaker = [];

fprintf('Finished initialization. \n')

%% perform actual experiment with recording !!
opt.isi = 400;
opt.filename = ['auditory_isi_' num2str(opt.isi) ];
opt.impedances = 0;
opt.test = 1;

    seq = accseq;
    seq = seq(1:60);
    sprintf('press <ENTER> to start the TESTRUN with ISI: %d', opt.isi)
    pause;
    play_auditory_oddball_ISI(seq, opt);
    sprintf('TESTRUN: how many did you count (TRUE NUMBER: %i, ISI %i)?\n',sum(seq), opt.isi)
for ii = 1:NRUNS
    seq = accseq;
    seq = [seq seq];
    sprintf('RUN %d press <RETURN> to proceed with the next trial, ISI: %d', ii, opt.isi)
    pause;
    if strcmp(func2str(acquire_func), 'acquire_bv')
        bvr_startrecording(opt.filename, 'impedance', 0)
    else
        signalServer_startrecoding(opt.filename)
    end
     
    pause(5);
    play_auditory_oddball_ISI(seq, opt);
    ppTrigger(255)
    if strcmp(func2str(acquire_func), 'acquire_bv'), bvr_sendcommand('stoprecording'), end;
    sprintf('how many did you count (TRUE NUMBER: %i, ISI %i)?\n',sum(seq), opt.isi)
end

