% Lade Workspace:
% reducerbox_64mcc_noEOG_EOGvu_replaces_PO6

%% INITILIZATION

bbci.classDef = {[1], [2], [3]; 'left', 'right', 'foot'};
all_classes= {'left', 'right', 'foot'};

DO_EMG = 0;

if DO_EMG

signalserver_impedance('server_config_EMG_LRF.xml')
pause
    start_signalserver('server_config_EMG_LRF_BUK.xml')
end
%% Standard Recordings: Augenbewegungen, Blinzeln, Entspannen mit Augen
%% auf/zu TEST

[seq, wav, opt]= setup_TOBI_games_MI_artifacts_demo('clstag', '');
fprintf('Press <RETURN> when ready to start ARTIFACT TEST measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt, 'test',1, 'useSignalServer', 0);

%% - Artefaktmessung: Augenbewegungen, Ruhemessung mit Augen auf und Augen zu
fprintf('\n\nArtifact recording.\n');
[seq, wav, opt]= setup_tobi_games_MI_artifacts('clstag', '');
fprintf('Press <RETURN> when ready to start ARTIFACT measurement.\n');
pause
if DO_EMG;    signalServer_startrecoding('EMG_arte_amAnfang');end;
stim_artifactMeasurement(seq, wav, opt,'filename', 'arte_amAnfang', 'useSignalServer', 0);
ppTrigger(254) % just to ensure that the recording is stopped (also for g-Tec)!

%% std auditory measurement
run_auditory_std_exp

%% Run1
%% no feedback!

nRuns= 3;
tag_list= {'LR', 'LF', 'FR'};
for ri= 2:nRuns,
    for ti= 1:length(tag_list),
        CLSTAG= tag_list{ti};
        fprintf('\n\n');fprintf(['NO FEEDBACK RUN:    please switch to the calibration mode ''' CLSTAG ''' \n' ]);
        inp = '';
        while ~strcmp(inp, 'start')
            sprintf('\n');
            inp = input('type ''start''  if you want to start the recording: ', 's');
        end
        
        fname = [bbci.fileNames{1} '_' CLSTAG];
%         if DO_EMG;    signalServer_startrecoding(['EMG_' fname]);end;
        bvr_startrecording(fname, 'impedances', 0); 
%         signalServer_startrecoding(fname)

        disp('start recording..')
        pause(3)
        fprintf(['\n START THE GAME IN CALIBRATION MODE  ''' CLSTAG ''' \n']);
        while ~strcmp(inp, 'next')
            sprintf('\n')
            inp = input('if calibration ended, type ''next''  if you want to FINISH THE RECORDING continue to the next run: ', 's');
        end
        ppTrigger(254) % just to ensure that the recording is stopped (also for g-Tec)!
        bvr_sendcommand('stoprecording');
    end
end

%% Train classifier on the No-feedback Runs
tag_list = {'LF'} %FOR TESTING
tag_list= {'LR', 'LF', 'FR'};
tag_list= {'LR', 'FR'};
tag_list = {'FR'}

tag_list = {'LR'}
for ti= 1:length(tag_list),
    bbci= bbci_default;
    CLSTAG= tag_list{ti};
    bbci.train_file = [bbci.subdir bbci.fileNames{1} '_' CLSTAG '*'];
%     bbci.train_file = [bbci.subdir 'imag_fb_bar_pcovmean_getFour_VPlh_FR'];
    
    bbci.save_name= [TODAY_DIR bbci.classyNames{1} '_' CLSTAG];
    bbci.adaptation.running= 1; %this only applies to cls(1) -->
    bbci.adaptation.policy = 'pcovmean_multi';

    bbci_bet_prepare
    bbci.adaptation.load_tmp_classifier =  0;
    bbci.adaptation.UC_mean = 0.075;
    bbci.adaptation.UC_pcov = 0.03;
    bbci.adaptation.verbose = 2;
    bbci.adaptation.ix_adaptive_cls = [1 2];
    bbci.adaptation.adaptation_ival = bbci.setup_opts.ival;
    bbci.setup_opts.do_LRP = 1;
    bbci.setup_opts.do_csp = 1;
    bbci.setup_opts.ival = {[1500 5800] [650 1400]};
%         bbci.setup_opts.ival = {[1000 2500] [450 1000]};
    bbci.setup_opts.band = {[10 12], [0.1 3]};%CSP/LRP (LRP is not really used ?!?)
    bbci.setup_opts.reject_opts = {'do_multipass' , 1 , 'whiskerperc' , 10, 'whiskerlength', 3 , 'do_relvar' , 1 , 'do_bandpass', 0};
     bbci.setup_opts.visu_ival = [-500 8000]
    
     bbci.setup_opts.clab = {{'*'}, {'*'} };%CSP/LRP

    bbci.setup_opts.do_laplace = 1;
% % 
%      clab16 = { 'FC3'    'FCz'    'FC4'    'C5'    'C3'    'C1'    'Cz'    'C2'    'C4'    'C6'    'CP5'    'CPz'    'CP6'    'P3'    'Pz'    'P4'};
%      bbci.setup_opts.clab = {clab16 clab16};
%     bbci.setup_opts.do_laplace = 0;
%      Cnt = proc_selectChannels(Cnt, clab16);
%     
    bbci_bet_analyze
    
    dum = bbci.classDef(1,[find(strcmp(bbci.classDef(2,:), bbci.classes(1))) find(strcmp(bbci.classDef(2,:), bbci.classes(2)))] );
    %     keyboard
    bbci.adaptation.mrk_start = dum;
    bbci.adaptation.mrk_end = [11 12 13 101:103 111:113];

    fprintf('Type ''bbci_bet_finish'' to save classifier and ''dbcont'' proceed.\n');
    keyboard
    close all

    %bbci_bet_finish %% only if you are really sure about that!
end
