%% Initialisierung
pause on % Pausen einschalten

blocks = 12;                %Anzahl der Blöcke in der Hauptmessung
targets = 10;               %Zieltöne pro Sequenz
interval_standard_1 = 1000; %Abstand der Töne in der ersten Standardmessung
interval_standard_2 = 225;  %Abstand der Töne in der zweiten Standardmessung
duration_standard_1 = 3;    %Dauer des Standard-Experiments1 in Minuten
duration_standard_2 = 2;    %Dauer des Standard-Experiments2 in Minuten
rate = 6;                   %Verhältnis Target zu Non-Target: 1/rate
Interval_main = 225;        %Abstand der Silben in der Hauptmessung


%init udp and pyff
bvr_sendcommand('viewsignals');
pause(2)
send_xmlcmd_udp('init', '127.0.0.1', 12345);
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'T9Speller','command','sendinit');
pause(3)
send_xmlcmd_udp('interaction-signal', 'i:spellerMode', false );
send_xmlcmd_udp('interaction-signal', 'i:simulate_sbj', false );
send_xmlcmd_udp('interaction-signal', 's:LOG_FILENAME', [TODAY_DIR '/calibration.log'] );

%% standard measurement
%bvr_checkparport;


disp('start standard measurement')
disp('EYES OPEN - check for fixation cross, press <ENTER> to start')
pause;
% Aufnahme mit geöffneten Auge 90s
bvr_startrecording(['eyesOpen_' VP_CODE], 'impedances', 1);
pause(90);
bvr_sendcommand('stoprecording');

% Aufnahme mit geschlossenen Augen
sprintf('\n \n \n \n EYES CLOSED, press <ENTER> to start')
pause;
bvr_startrecording(['eyesClosed_' VP_CODE], 'impedances', 0);

pause(90);

bvr_sendcommand('stoprecording');




%%

% 2. Teil: Standardmessung
%- 2x Kurzmessung: 2 Pieptöne(Oddball, 1:5), erst 1000ms Abstand, zweite
%225ms(wie im Experiment)
opt = [];
opt.toneDuration = 40;
opt.speakerSelected = [6 2 4 1 5 3];
opt.language = 'german';

%setup_spatialbci_GLOBAL

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
pause;
% startMeasure(sequenz);    %Beginn der Messung, Speichern der Daten in 'file'
% eegfile_saveMatlab(file, dat, mrk); % Daten speicherns
        opt.isi = 1000;
        opt.filename = ['auditory_isi_' VP_CODE '_' num2str(opt.isi) '_std'];
        opt.impedances = 0;
        sprintf('press <RETURN> to proceed with the standard Auditory Oddball Experiment with ISI: %d', opt.isi)        
        pause;
        play_auditory_oddball_ISI(sequenz, opt);
        sprintf('how many did you count (TRUE NUMBER: %i, ISI %i)?\n',sum(sequenz), opt.isi)

sequenz = standard_oddball(rate,interval_standard_2,duration_standard_2);
sequenz  = sequenz - 1;
        opt.isi = 225;
        opt.filename = ['auditory_isi_' VP_CODE '_' num2str(opt.isi) '_std'];
        opt.impedances = 0;
        sprintf('press <RETURN> to proceed with the standard Auditory Oddball Experiment with ISI: %d', opt.isi)        
        pause;
        play_auditory_oddball_ISI(sequenz, opt);
        sprintf('how many did you count (TRUE NUMBER: %i, ISI %i)?\n',sum(sequenz), opt.isi)

        
        
%% 3. Teil: Testmessung
fprintf('Now we start the main measurement... press <ENTER> to proceed')
pause;

for i = 1:1
    sbj_counts = 0;
    target_seq = rdm_seq(1,1);
    
%    bvr_startrecording(['T9calibration_wharp_' VP_CODE], 'impedances', 0);
    send_xmlcmd_udp('interaction-signal', 'i:keysToSpell' , target_seq);
    
    my_seq = rdm_seq(targets,0);
    send_xmlcmd_udp('interaction-signal', 'i:GLOBAL_SEQ' , my_seq);
    
    pause(1)
    send_xmlcmd_udp('interaction-signal', 'command', 'play');
    
    for trial=1:9
        %Messung starten, passenden Trial auswählen
        % send the number of counts and thereby go to the next trial
            numTargets = sum(my_seq == target_seq(trial) );
            fprintf('True number: %d \n!',numTargets);
            %catch counts of the trials!!
            sbj_counts = input('enter the counted number (when block completed)', 's');
            if strmatch(sbj_counts,'end')
                break;      
            else
              sbj_counts = str2num(sbj_counts);
              if ~isempty(sbj_counts)
                 my_seq = rdm_seq(targets,0); %generate new seq for next trial
                 send_xmlcmd_udp('interaction-signal', 'i:GLOBAL_SEQ' , my_seq);
                 send_xmlcmd_udp('interaction-signal', 'i:numCounts' , sbj_counts);
              end
            end
    end
%    bvr_sendcommand('stoprecording');
     fprintf('Hauptmessung: Trial %d  beendet! Press <ENTER> to start the next trial!',trial);
    pause;
end   
        
%% 4. Teil: Hauptmessung
fprintf('Now we start the main measurement... press <ENTER> to proceed')
pause;

for i = 1:blocks
    send_xmlcmd_udp('interaction-signal', 's:_feedback', 'T9Speller','command','sendinit');
    pause(3)
    send_xmlcmd_udp('interaction-signal', 'i:spellerMode', false );
    send_xmlcmd_udp('interaction-signal', 'i:simulate_sbj', false );
    send_xmlcmd_udp('interaction-signal', 'i:MAX_NONMARKER_SEQ', 1);
    send_xmlcmd_udp('interaction-signal', 's:LOG_FILENAME', [TODAY_DIR '/calibration.log'] );

    sbj_counts = 0;
    target_seq = rdm_seq(1,0);
    
    bvr_startrecording(['T9calibration_wharp_' VP_CODE], 'impedances', 0);
    send_xmlcmd_udp('interaction-signal', 'i:keysToSpell' , target_seq);
    
    my_seq = rdm_seq(targets,0);
    send_xmlcmd_udp('interaction-signal', 'i:GLOBAL_SEQ' , my_seq);
    
    fprintf('Press <ENTER> to start Start Block %d !',i);
    pause;
    send_xmlcmd_udp('interaction-signal', 'command', 'play');
    
   for trial=1:9
        %Messung starten, passenden Trial auswählen
        % send the number of counts and thereby go to the next trial
            numTargets = sum(my_seq == target_seq(trial) );
            fprintf('True number: %d \n!',numTargets);
            %catch counts of the trials!!
            sbj_counts = input('enter the counted number (when block completed)', 's');
            if strmatch(sbj_counts,'end')
                break;      
            else
              sbj_counts = str2num(sbj_counts);
              if ~isempty(sbj_counts)
                 my_seq = rdm_seq(targets,0); %generate new seq for next trial
                 send_xmlcmd_udp('interaction-signal', 'i:GLOBAL_SEQ' , my_seq);
                 send_xmlcmd_udp('interaction-signal', 'i:numCounts' , sbj_counts);
              end
            end
    end
    bvr_sendcommand('stoprecording');
    send_xmlcmd_udp('interaction-signal', 'command', 'quit');
     fprintf('Block %d  beendet! Press <ENTER> to start the next trial!',trial);
    pause;
end    





