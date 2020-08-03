%% Initialisierung
pause on % Pausen einschalten

nblocks = 9;                %Anzahl der Blöcke in der Hauptmessung
ncond = 5;

targets = 10;               %Zieltöne pro Sequenz
interval_standard_1 = 1000; %Abstand der Töne in der ersten Standardmessung
interval_standard_2 = 130;  %Abstand der Töne in der zweiten Standardmessung
duration_standard_1 = 3;    %Dauer des Standard-Experiments1 in Minuten
duration_standard_2 = 2;    %Dauer des Standard-Experiments2 in Minuten
rate = 5;                   %Verhältnis Target zu Non-Target: 1/rate


%init udp and pyff
bvr_sendcommand('viewsignals');
pause(2)
send_xmlcmd_udp('init', '127.0.0.1', 12345);

%%
system('cmd /C "e: & cd \ & cd svn\bbci\python\pyff\src\Feedbacks\Auditory_stimulus_screening & python TrialPresentation.py &');



%% while setting up the cap, present stimuli
fprintf('Familiarise with sounds and conditions... press <ENTER> to proceed\n')
pause;
cc = 0;
while 1
    sbj_counts = 0;
    cc = cc+1;
    if cc == ncond + 1
        cc = 1
    end
%    bvr_startrecording(['T9calibration_wharp_' VP_CODE], 'impedances', 0);
    send_xmlcmd_udp('interaction-signal', 's:_feedback', 'Auditory_stimulus_screening','command','sendinit');    
    pause(3)
    send_xmlcmd_udp('interaction-signal', 'i:loadCond' , cc);  
    send_xmlcmd_udp('interaction-signal', 'i:N_MARKER_SEQ' , 7);
    send_xmlcmd_udp('interaction-signal', 'i:ISI' , 400);    
    send_xmlcmd_udp('interaction-signal', 'i:simulate_sbj' , true); %trials are not paused for 'ask4counts'
    
    pause(1)
    send_xmlcmd_udp('interaction-signal', 'command', 'play');
    
    fprintf('Presentation of Condition %d! Press <ENTER> to start next condition !',cc);
    pause
    fprintf('    Sure ?!? \n \n');
    pause
    send_xmlcmd_udp('interaction-signal', 'command', 'quit');    
    pause(2)
end   



%% standard measurement
%bvr_checkparport;

disp('start standard measurement')
disp('EYES OPEN - check for fixation cross, press <ENTER> to start')
pause;
% Aufnahme mit geöffneten Auge 90s
bvr_startrecording(['eyesOpen_' VP_CODE], 'impedances', 1);
pause(50);
bvr_sendcommand('stoprecording');

% Aufnahme mit geschlossenen Augen
sprintf('\n \n \n \n EYES CLOSED, press <ENTER> to start')
pause;
bvr_startrecording(['eyesClosed_' VP_CODE], 'impedances', 1);

pause(50);

bvr_sendcommand('stoprecording');

disp('start standard measurement')
disp('EYES OPEN - check for fixation cross, press <ENTER> to start')
pause;
% Aufnahme mit geöffneten Auge 90s
bvr_startrecording(['eyesOpen_' VP_CODE], 'impedances', 0);
pause(50);
bvr_sendcommand('stoprecording');

% Aufnahme mit geschlossenen Augen
sprintf('\n \n \n \n EYES CLOSED, press <ENTER> to start')
pause;
bvr_startrecording(['eyesClosed_' VP_CODE], 'impedances', 0);

pause(50);

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

sequenz = standard_oddball(9,interval_standard_2,duration_standard_2);
sequenz  = sequenz - 1;
        opt.isi = 130;
        opt.filename = ['auditory_isi_' VP_CODE '_' num2str(opt.isi) '_std'];
        opt.impedances = 0;
        sprintf('press <RETURN> to proceed with the standard Auditory Oddball Experiment with ISI: %d', opt.isi)        
        pause;
        play_auditory_oddball_ISI(sequenz, opt);
        sprintf('how many did you count (TRUE NUMBER: %i, ISI %i)?\n',sum(sequenz), opt.isi)
     
        
        
%% 4. Teil: Hauptmessung
fprintf('Now we start the REAL measurement... press <ENTER> to proceed\n')
pause;

cond_in_block = [   1 4 3 2 5;
    2 5 4 3 1;
    3 1 5 4 2;
    4 2 1 5 3;
    5 3 2 1 4;
    3 5 4 1 2;
    1 4 3 2 5;
    2 5 4 3 1;
    3 1 5 4 2;
    ]

for i = 1:nblocks
    if or(or(i == 1, i == 4), i == 7)
       targets = {randperm(9) randperm(9) randperm(9) randperm(9) randperm(9)};
    end
    targets
    for j = 1:ncond
        cc = cond_in_block(i,j)
        send_xmlcmd_udp('interaction-signal', 's:_feedback', 'Auditory_stimulus_screening','command','sendinit');    
        pause(2)
        send_xmlcmd_udp('interaction-signal', 'i:loadCond' , cc);   
        
        send_xmlcmd_udp('interaction-signal', 's:LOG_FILENAME', [TODAY_DIR '/Log_' num2str(i) '_' num2str(j) '_cond_' num2str(cc) '.log'] );
        ss =  targets{cc}(1:3);
        targets{cc}(1:3) = [];
        ss
        send_xmlcmd_udp('interaction-signal', 'i:keysToSpell' , ss);    
        
        sbj_counts = 0;

        bvr_startrecording(['Audit_stim_screening_cond_' num2str(cc) '_' VP_CODE], 'impedances', 0);

        fprintf('Type <go> to start Start Block %d ! \n \n',i);

        inp = '';
%         keyboard
        while ~strcmp(inp, 'go')
            inp = input('waiting...', 's');
        end
        
        send_xmlcmd_udp('interaction-signal', 'command', 'play');

       for trial=1:3
            %Messung starten, passenden Trial auswählen
            % send the number of counts and thereby go to the next trial

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
        bvr_sendcommand('stoprecording');
        send_xmlcmd_udp('interaction-signal', 'command', 'quit');
         fprintf('condition %d  beendet! Press <ENTER> to start the next condition!\n',cc);
        pause;
    end
end    





