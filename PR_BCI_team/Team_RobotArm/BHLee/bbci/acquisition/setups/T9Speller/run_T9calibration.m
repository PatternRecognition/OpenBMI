%% setup everything
% @JohannesHoehne 

bvr_sendcommand('viewsignals');
pause(2)
send_xmlcmd_udp('init', '127.0.0.1', 12345);

%testrun
fprintf('Press <RET> to start the standard test-runs.\n');
fprintf('EXPLAIN THE TASK: count the infrequent (high) tone.\n \n');
fprintf('MENTION THAT THE VOICE SHOULD BE NEGLECTED: OPEN EYES AND FOCUSS THE FIXATION CROSS\n');
pause


%% TESTRUN
N=200;  %FORTESTING  % usual on 150
iterations = 2;
clear opt;

opt.toneDuration = 40;
opt.speakerSelected = [6 2 4 1 5 3];
opt.language = 'german';

%setup_spatialbci_GLOBAL

opt.isi_jitter = 0; % defines jitter in ISI

opt.itType = 'fixed';
opt.mode = 'copy';
opt.application = 'TRAIN';

opt.countdown = 0;
opt.repeatTarget = 3;

opt.perc_dev = 0.2%111;%20/100;
opt.avoid_dev_repetitions = 1;
opt.require_response = 0;
opt.isi = 1000; %same as T9Speller!
opt.fixation = 1;
opt.filename = 'oddballStandardMessung';
%opt.speech_intro = '';
opt.fixation =1;
opt.fs = 44100;
opt.cue_std = stimutil_generateTone(500, 'harmonics', 7, 'duration', 50, 'pan', 1, 'fs', opt.fs);
opt.cue_std = opt.cue_std*.25;
opt.cue_dev = stimutil_generateTone(1000, 'harmonics', 7, 'duration', 50, 'pan', 1, 'fs', opt.fs);
opt.cue_dev = opt.cue_dev*.25;


for i = 1:iterations,
  stim_oddballAuditory(N, opt);
  stimutil_waitForInput('phrase', 'go', 'msg', 'When run has finished, type go to proceed "go<RET>".');
end
fprintf('\n \n \nplease close the Window with the fixation cross! \n \n')
fprintf('------------------- \n \n \n')
%% Calibration runs

fprintf('\n \n The subject should get used to the fast sequence of sounds. \n As a little example we we make some test-trials.\n');
fprintf('Start with <ENT> and cancel this run with Ctrl-C in the Console \n when subject has got used to the speed !!\n');
fprintf('Remember: EEG data is NOT recorded for this run!!! \n')
pause
system('cmd /C "E: & cd \ & cd svn\bbci\python\pyff\src\Feedbacks\T9Speller & python T9Speller.py &');
fprintf('when finished, press <ENT> \n \n')
pause

fprintf('Now, CALIBRATION runs are about to start (EEG signal is recorded), press <ENT> to start \n!')
pause
numCalibRuns = 3 %4

i=3
% send_xmlcmd_udp('fc-signal', 's:TODAY_DIR','', 's:VP_CODE','', 's:BASENAME','');
for i=1:numCalibRuns
    sbj_counts = 0;
    bvr_startrecording(['T9SpellerCalibration' VP_CODE], 'impedances', 0); 
    send_xmlcmd_udp('interaction-signal', 's:_feedback', 'T9Speller','command','sendinit');
    pause(5);
    send_xmlcmd_udp('interaction-signal', 'i:spellerMode', false );
    send_xmlcmd_udp('interaction-signal', 'i:simulate_sbj', false );
    send_xmlcmd_udp('interaction-signal', 's:LOG_FILENAME', [tmpLogDir 'calibration' num2str(i) '.log'] );
    
    pause(1)
    send_xmlcmd_udp('interaction-signal', 'command', 'play');

    while isnumeric(sbj_counts)
        %catch counts of the trials!!
        sbj_counts = input('enter the counted number (when block completed)', 's');
        if strmatch(sbj_counts,'end')
            break;      
        else
          sbj_counts = str2num(sbj_counts);
          if ~isempty(sbj_counts)
            send_xmlcmd_udp('interaction-signal', 'i:numCounts' , sbj_counts);
          end
        end
    end
    send_xmlcmd_udp('interaction-signal', 'command', 'quit');
    bvr_sendcommand('stoprecording');
    fprintf('Press <RET> to continue with the next calibration run.\n');
    pause
end


%% learn the classifier
fprintf('Press <RET> to start training the classifier.\n');
pause

setup_T9_online;
bbci_bet_prepare;
bbci_bet_analyze;

fprintf('Press <RET> to finish the calibration.\n');
pause
% only when analyzes satisfiable
bbci_bet_finish;

fprintf('type run_online_spelling to continue with online Spelling!\n');

