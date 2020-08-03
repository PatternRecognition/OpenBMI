%% initialization

%init udp and pyff
bvr_sendcommand('viewsignals');
pause(2)
send_xmlcmd_udp('init', '127.0.0.1', 12345);
disp('parameters set')

%% -------- Kappe praeparieren  -------
%% Toene abspielen

%Lautstaerke
system('cmd /C "e: & cd \ & cd svn\bbci\python\pyff\src\Feedbacks\Auditory_stimulus_screening & python TrialPresentation.py &');

%% Stimuli zaehlen lassen
% TODO: Putzen!!!

fprintf('Familiarise with sounds and conditions... press <ENTER> to proceed\n')
pause;
fprintf('los...')

SOA = 200; % erst 400, danach 200


send_xmlcmd_udp('interaction-signal', 's:_feedback', 'Auditory_stimulus_screening','command','sendinit');
pause(3)
send_xmlcmd_udp('interaction-signal', 's:loadParafile' , 'paraSpatialSOAintervals');
send_xmlcmd_udp('interaction-signal', 'i:N_MARKER_SEQ' , 7);
send_xmlcmd_udp('interaction-signal', 'i:ISI' , SOA);
send_xmlcmd_udp('interaction-signal', 'i:simulate_sbj' , true); %trials are not paused for 'ask4counts'

send_xmlcmd_udp('interaction-signal', 'i:keysToSpell' , [randperm(9) randperm(9)] );

pause(1)
send_xmlcmd_udp('interaction-signal', 'command', 'play');

fprintf('Press >ENTER> to stop the presentation of stimuli\n');
pause
fprintf('    Sure ?!? \n \n');
pause
send_xmlcmd_udp('interaction-signal', 'command', 'quit');






%% ------ Ruhemessungen ----------

eyesOpenMarker = 1;
eyesClosedMarker = 2;
block_time = 60; % Blockzeit, in Sekunden
disp('parameters set')

%% Augen auf/zu, Audio aus


disp('start standard measurement')
disp('Augen auf/zu, Audio aus, press <ENTER> to start')
pause;

clear wavfile
wavfile = wavread('ansage_4min_10s.wav');

bvr_startrecording(['resting_eyesOpenClosed_audioOff_' VP_CODE], 'impedances', 0);
pause(2);
wavplay(wavfile, 44100,'async'); % Audio anmachen!!! Und Matlab nicht blockiert!!!

disp('Augen offen')
ppTrigger(eyesOpenMarker);
pause(block_time-3);
display('');disp('Noch 3 Sekunden!!!') 
pause(3);

disp('Augen schliessen')
ppTrigger(eyesClosedMarker);
pause(block_time-3);
disp('\n Noch 3 Sekunden!!!') 
pause(3);

disp('Augen offen')
ppTrigger(eyesOpenMarker);
pause(block_time-3);
display('');disp('Noch 3 Sekunden!!!') 
pause(3);

disp('Augen schliessen')
ppTrigger(eyesClosedMarker);
pause(block_time);

disp('ENDE')
pause(3)
bvr_sendcommand('stoprecording');

clear playsnd;
% audio aus oder zuende

%% Augen auf/zu, Audio an, Inhalt wiedergeben


disp('Augen auf/zu, Audio An, Inhalt wiedergeben, press <ENTER> to start')
pause;
clear wavfile
wavfile = wavread('kafka_ein_traum_4min_10s_mit_ansage.wav');

bvr_startrecording(['resting_eyesOpenClosed_audioOn_taskSummary_' VP_CODE], 'impedances', 0);
pause(2);
wavplay(wavfile, 44100,'async'); % Audio anmachen!!! Und Matlab nicht blockiert!!!


disp('Augen offen')
ppTrigger(eyesOpenMarker);
pause(block_time-3);
display('');disp('Noch 3 Sekunden!!!') 
pause(3);

disp('Augen schliessen')
ppTrigger(eyesClosedMarker);
pause(block_time-3);
display('');disp('Noch 3 Sekunden!!!') 
pause(3);

disp('Augen offen')
ppTrigger(eyesOpenMarker);
pause(block_time-3);
display('');disp('Noch 3 Sekunden!!!') 
pause(3);

disp('Augen schliessen')
ppTrigger(eyesClosedMarker);
pause(block_time);

disp('ENDE')
pause(3)
bvr_sendcommand('stoprecording');

clear playsnd;
% audio aus oder zuende

%% Augen auf/zu, Audio an, Woerter zaehlen

% Bitte Aufgabe von der Person nochmals wiederholen lassen! 

disp('Augen auf/zu, Audio An, Woerter zaehlen, press <ENTER> to start')
pause;

clear wavfile
wavfile = wavread('kafka_ein_traum_4min_10s_mit_ansage.wav');

bvr_startrecording(['resting_eyesOpenClosed_audioOn_taskWordCount_' VP_CODE], 'impedances', 0);
pause(2);
wavplay(wavfile, 44100,'async'); % Audio anmachen!!! Und Matlab nicht blockiert!!!

disp('Augen offen')
ppTrigger(eyesOpenMarker);
pause(block_time-3);
display('');disp('Noch 3 Sekunden!!!') 
pause(3);

disp('Augen schliessen')
ppTrigger(eyesClosedMarker);
pause(block_time-3);
display('');disp('Noch 3 Sekunden!!!') 
pause(3);

disp('Augen offen')
ppTrigger(eyesOpenMarker);
pause(block_time-3);
display('');disp('Noch 3 Sekunden!!!') 
pause(3);

disp('Augen schliessen')
ppTrigger(eyesClosedMarker);
pause(block_time);

disp('ENDE')
pause(3)
bvr_sendcommand('stoprecording');

% audio aus oder zuende
clear playsnd;

%% Standard Oddball

% Bitte Aufgabe von der Person nochmals wiederholen lassen! 
%Lautstaerke
% hohen Ton zählen lassen

interval_standard_1 = 1000; %Abstand der Tï¿½ne in der ersten Standardmessung
duration_standard_1 = 3;    %Dauer des Standard-Experiments1 in Minuten
rate = 5;                   %Verhï¿½ltnis Target zu Non-Target: 1/rate


opt = [];
opt.toneDuration = 40;
opt.speakerSelected = [6 2 4 1 5 3];
opt.language = 'german';

opt.isi_jitter = 0; % defines jitter in ISI
opt.itType = 'fixed';
opt.mode = 'copy';
opt.application = 'TRAIN';
opt.fixation = 1;
opt.filename = 'auditory_isi';
opt.speech_intro = '';
opt.fixation = 1;
opt.fs = 44100;
opt.cue_std = stimutil_generateTone(500, 'harmonics', 7, 'duration', 50, 'pan', 1, 'fs', opt.fs);
opt.cue_std = opt.cue_std*.25;
opt.cue_dev = stimutil_generateTone(1000, 'harmonics', 7, 'duration', 50, 'pan', 1, 'fs', opt.fs);
opt.cue_dev = opt.cue_dev*.25;


sequenz = standard_oddball(rate,interval_standard_1,duration_standard_1);
sequenz  = sequenz - 1;

fprintf('Standardmessung 1 starten');
opt.isi = 1000;
opt.filename = ['auditory_isi_' VP_CODE '_' num2str(opt.isi) '_std'];
opt.impedances = 0;
opt.test = 0; % test = 1 means no recording!!!
sprintf('press <RETURN> to proceed with the standard Auditory Oddball Experiment with ISI: %d', opt.isi)
pause;
play_auditory_oddball_ISI(sequenz, opt);
sprintf('how many did you count (TRUE NUMBER: %i, ISI %i)?\n',sum(sequenz), opt.isi)



%% ------ Hauptmessung ---------

%% Sequenzen generieren oder nachladen
block_info1 = generate_experiment_blocks(9, 3.5, 4, 5.5);
block_info2 = generate_experiment_blocks(6, 3.5, 4, 5.5);
fname = [TODAY_DIR 'block_info.mat'];
if ~exist(fname, 'file')
    save(fname, 'block_info*')
    disp('saved block_info into file!')
else 
    load(fname)
    disp('loaded block_info from file!')
end
%% PYFF Feedback init
fracCatchTrials = 0.09
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'Auditory_stimulus_screening','command','sendinit');
pause(2)
send_xmlcmd_udp('interaction-signal', 's:loadParafile' , 'paraSpatialSOAintervals');

disp('Init and parameters set')
%% Messung Teil 1 starten

%Lautstaerke
% Bitte Aufgabe von der Person nochmals wiederholen lassen! 

disp('ACHTUNG! bitte das Python Fenster SOFORT einmal anklicken!!! \n \n')

for block_idx=1:5
    for trial_idx=1:9
        target_idx = block_info1{block_idx,1}(trial_idx);
        SOA = block_info1{block_idx,2}(trial_idx);
        cue_delay = block_info1{block_idx,3}(trial_idx);
        disp(sprintf('block idx = %d, trial idx = %d, SOA = %d', block_idx, trial_idx, SOA)) 
        display('Press <ENTER> to start trial')
        pause;
        
        % insert catch trials here
        if rand(1) < fracCatchTrials
            disp('executing CATCH trial!')
            execute_trial(SOA, target_idx, 3.5, VP_CODE, 1, 15);
            disp(sprintf('block idx = %d, trial idx = %d, SOA = %d', block_idx, trial_idx, SOA)) 
        
            display('Press <ENTER> to start trial')
            pause;            
        end
        
        % execute regular trial
        execute_trial(SOA, target_idx, cue_delay, VP_CODE, 0, 15);
    end
    fprintf('Block %d finished. Press <ENTER> to continue. Pause?\n', block_idx)
    pause;
end
%% Messung Teil 2 starten

% Bitte Aufgabe von der Person nochmals wiederholen lassen! 
% die Mitteltoene fehlen

disp('ACHTUNG! bitte das Python Fenster SOFORT einmal anklicken!!! \n \n')

for block_idx=2:5
    for trial_idx=1:6
        target_idx = block_info2{block_idx,1}(trial_idx);
        SOA = block_info2{block_idx,2}(trial_idx);
        cue_delay = block_info2{block_idx,3}(trial_idx);
        disp(sprintf('block idx = %d, trial idx = %d, SOA = %d', block_idx, trial_idx, SOA)) 
        display('Press <ENTER> to start trial')
        pause;
        
        % insert catch trials here
        if rand(1) < fracCatchTrials
            disp('executing CATCH trial!')
            execute_trial(SOA, target_idx, 3.5, VP_CODE, 1, 5);
            disp(sprintf('block idx = %d, trial idx = %d, SOA = %d', block_idx, trial_idx, SOA))             
            
            display('Press <ENTER> to start trial')
            pause;
        end
        
        % execute regular trial
        execute_trial(SOA, target_idx, cue_delay, VP_CODE, 0, 5);
    end
    fprintf('Block %d finished. Press <ENTER> to continue. Pause?\n', block_idx)
    pause;
end

%% ------ ENDE -----------------------------

%% ------ SSCA stuff ------------------------

%% parameters

T = 60*0.5; % total time in seconds
ss_freq = 40; % steady state frequency in Hertz
sound_fs = 44100; % sampling frequency of the wave sound
modulation_cutoff_freq = 0.025;

start_marker = 251;
end_marker = 254;

%% generate sounds

fname = [TODAY_DIR 'ssca_sound_data.mat'];
% if ~exist(fname, 'file')
    long_sound = randn(1,T*sound_fs);
    N = length(long_sound);
    steady_state_modulation = 0.5 + 0.5*sin((1:N)*2*pi*ss_freq*T/N);
    ss_sound = long_sound.*steady_state_modulation;

    modulation_db = create_slow_random_signal(T, sound_fs, modulation_cutoff_freq)';
    modulation_db = modulation_db(2:end)';
    modulation_db = modulation_db-min(modulation_db);
    modulation_db = modulation_db/max(modulation_db);
    modulation_db = 20*(modulation_db - 0.5);
    modulation = 10.^(modulation_db/10);

    modulated_ss_sound = ss_sound .* modulation;
    
    save(fname, 'ss_sound', 'modulation_db', 'modulation', 'modulated_ss_sound')
    disp('saved sound data into file!')
% else 
%     load(fname)
%     disp('loaded sound_data from file!')
% end



%% plot amplitude modulation
N = sound_fs * T;
time = (1:N)/sound_fs;
figure
rows = 2;
cols = 1;
subplot(rows, cols, 1)
plot(time, modulation_db)
title('Amplitude modulation of steady-state sound in dB units')
xlabel('time in seconds')
ylabel('dB')
subplot(rows, cols, 2)
plot(time, modulation)
title('Amplitude modulation of steady-state sound absolute units')
xlabel('time in seconds')
ylabel('a.u.')


%% play sound for testing

wavplay(modulated_ss_sound,  sound_fs, 'async')
%clear playsnd;


%% play the sound with eeg recording

bvr_startrecording(['SSAEP_eyesOpen_' VP_CODE], 'impedances', 0);
pause(3);
ppTrigger(start_marker)
wavplay(modulated_ss_sound,  sound_fs, 'sync')
ppTrigger(end_marker)
pause(3)
bvr_sendcommand('stoprecording');

display('FINISHED')


