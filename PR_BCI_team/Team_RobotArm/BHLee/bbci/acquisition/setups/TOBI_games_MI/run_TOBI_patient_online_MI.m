%% TODOs
% fnamePreviousSessions update
fname_auditory_MI = ['imag_LR_auditory_' VP_CODE];

% fnamePreviousSessions = {'D:\data\bbciRaw\VPfbd_11_08_03/imag_calib_bar_getFour_VPfbd_LR*',...
%  fnamePreviousSessions =   {'D:\data\bbciRaw\VPfbd_11_08_15/imag_fb_bar_pcovmean_getFour_VPfbd_endOfTrialFB_LR*'};
% % classyNamePreviousSession = 'VPfbd_11_08_03\INITIAL_bbci_classifier_csp_LRP_16chans_pcovmean_multi_VPfbd_LR.mat'; 
% classyNamePreviousSession = 'VPfbd_11_08_03\VPfbd_11_08_15\INITIAL2_bbci_classifier_csp_LRP_16chans_pmean_multi_VPfbd_LR'; 

%Klaus
% initial_classifier = 'D:\data\bbciRaw\VPfbd_11_08_03\INITIAL_bbci_classifier_csp_LRP_16chans_pcovmean_multi_VPfbd_LR.mat';
% initial_classifier =
% 'D:\data\bbciRaw/VPfbe_11_09_05\INITIAL3_bbci_classifier_csp_LRP_16chans_pcovmean_multi_VPfbe_FR.mat'
% initial_classifier = 'D:\data\bbciRaw/VPfbd_11_09_05\INITIAL5_bbci_classifier_csp_LRP_16chans_pcovmean_multi_VPfbd_LR.mat'
% initial_classifier = 'd:\data\bbciRaw/VPfbe_11_09_05\INITIAL5_bbci_classifier_csp_LRP_16chans_pcovmean_multi_VPfbe_FR.mat'


% initial_classifier = 'd:\data\bbciRaw/VPfbe_12_02_03\bbci_classifier_csp_LRP_16chans_pcovmean_multi_VPfbe_LR';
% initial_classifier = 'd:\data\bbciRaw/VPfbf_12_02_07\INITIAL_bbci_classifier_csp_LRP_16chans_pcovmean_multi_VPfbf_LR';
% initial_classifier = 'd:\data\bbciRaw/VPfbf_12_02_08\INITIAL2_bbci_classifier_csp_LRP_16chans_pcovmean_multi_VPfbf_LR';

%CSP mit beta rebound
% initial_classifier = 'd:\data\bbciRaw/VPfbf_12_02_08\INITIAL_beta__14_21__bbci_classifier_csp_LRP_16chans_pcovmean_multi_VPfbf_LR.mat';
% initial_classifier = 'd:\data\bbciRaw/VPfbf_12_02_10\INITIAL4_bbci_classifier_csp_LRP_16chans_pcovmean_multi_VPfbf_LR_beta-rebound002.mat';
initial_classifier = 'd:\data\bbciRaw/VPfbf_12_03_23\INITIAL5_bbci_classifier_csp_LRP_16chans_pcovmean_multi_VPfbf_LR.mat';

% fnamePreviousSessions = {'D:\data\bbciRaw/VPfbd_11_09_06\imag_fb_bar*';}
% fnamePreviousSessions = {'d:\data\bbciRaw/VPfbe_12_02_03\imag_fb_bar*';};

fnamePreviousSessions = {'d:\data\bbciRaw/VPfbf_12_02_07\imag_fb_bar*';};


initial_classifier = 'd:\data\bbciRaw/VPfbf_12_03_23\INITIAL5b_bbci_classifier_csp_LRP_16chans_pcovmean_multi_VPfbf_LR.mat'


% %MATZE
% initial_classifier = 'D:\data\bbciRaw/VPfbe_11_08_16\INITIAL_bbci_classifier_csp_LRP_16chans_pcovmean_multi_VPfbe_FR.mat';
% initial_classifier = 'D:\data\bbciRaw/VPfbe_11_08_17\INITIAL2_bbci_classifier_csp_LRP_16chans_pcovmean_multi_VPfbe_FR.mat';
% 
% fnamePreviousSessions =   {'D:\data\bbciRaw\VPfbe_11_08_17/imag*'};
% fnamePreviousSessions =   {'D:\data\bbciRaw\VPfbe_11_08_16/imag_calib_bar_getFour_VPfbe_FR*' 'D:\data\bbciRaw\VPfbe_11_08_17/imag*'};

fname_auditory_MI = ['imag_LR_auditory_' VP_CODE];

%% Record Impedances
signalserver_impedance('server_config_MI_BUK.xml')

%% INITILIZATION
% copyfile(initial_classifier, TODAY_DIR)

input('please close the impedance software')

start_signalserver('server_config_MI_BUK.xml')

input('please DEBLOCK the parallel port!');
input('please CHECK if the SignalServer is running properly!');
bbci.GAME = 'getFour';

%Klaus
% subject specific CLSTAG !!!
% CLSTAG = 'LR';
% bbci.classDef = {[1], [2] ; 'left', 'right'};
% all_classes= {'left', 'right'};

% Matze
% CLSTAG = 'FR';
% bbci.classDef = {[2], [3] ; 'right', 'foot'};
% all_classes= {'right', 'foot'};

% Andreas
CLSTAG = 'LR';
bbci.classDef = {[1], [2] ; 'left', 'right'};
all_classes= {'left', 'right'};


%% - Artefaktmessung: Augenbewegungen, Ruhemessung mit Augen auf und Augen zu
fprintf('\n\nArtifact recording.\n');
[seq, wav, opt]= setup_tobi_games_MI_artifacts_quick('clstag', '');
fprintf('Press <RETURN> when ready to start ARTIFACT measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt,'filename', 'arte_amAnfang', 'useSignalServer', 1);

%% - Artefaktmessung: Augenbewegungen, Ruhemessung mit Augen auf und Augen zu
fprintf('\n\nArtifact recording.\n');
[seq, wav, opt]= setup_tobi_games_MI_artifacts_LRfast('clstag', '');
fprintf('Press <RETURN> when ready to start ARTIFACT measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt,'filename', 'LR_eyesFast_amAnfang', 'useSignalServer', 1);


%% MOTOR IMAGINARY WITH AUDITORY CUES AND CLOSED EYES 
fprintf('\n\MOTOR IMAGINARY WITH AUDITORY CUES AND CLOSED EYES .\n');
[seq, wav, opt]= auditory_MI_via_artifacts('clstag', 'LR');
fprintf('Press <RETURN> when ready to start MI experiment.\n');
pause

signalServer_startrecoding(fname_auditory_MI);
stim_artifactMeasurement(seq, wav, opt,'filename', 'auditory_MI', 'test', 1, 'useSignalServer', 1);
ppTrigger(254); %stop recording


%% std auditory measurement
run_auditory_std_exp


%% start classifier of last session and set bias/gradient

cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d]; ', VP_CODE, TODAY_DIR, VP_SCREEN);
cmd_bbci= ['dbstop if error ; bbci_bet_apply(''' initial_classifier ''');'];
%cmd_bbci= ['dbstop if error ; bbci_bet_apply(''' classyNamePreviousSession
%''');']; 
system(['matlab -nosplash -r "' cmd_init 'setup_tobi_games_MI; ' cmd_bbci '; exit &']);

%% Run1a
%% show only feedback at the END OF TRIAL!
nRuns= 1;
for ri= 1:nRuns,
    sprintf('\n\n');disp(['Start the software in mode '''  CLSTAG ''' and set showBars = false !!!'])
    inp = '';
    while ~strcmp(inp, 'start')
        sprintf('\n')
        inp = input('type ''start'' if you want to start the recording: ', 's');
    end
    fname = [bbci.fileNames{2} '_' CLSTAG];

    signalServer_startrecoding(fname)
    %DEBUG with brother
%     bvr_startrecording(fname)

    sprintf('\n\n');disp(['PCOVMEAN - please start the calibration in the game, mode ''' CLSTAG ''' '])
    inp = '';
    while ~strcmp(inp, 'next')
        sprintf('\n')
        inp = input('type ''next  if you want to continue to the next run: ', 's');
    end
    ppTrigger(254) % just to ensure that the recording is stopped!
end

 input('stop the classifier!');

%% Train classifier on the current Runs 1a plus the old sessions
bbci.train_file = {[bbci.subdir bbci.fileNames{2} '_' CLSTAG '*']}
bbci.train_file = {[bbci.subdir 'imag_fb_bar_pcovmean_getFour_VPfbf*']}
% fnamePreviousSessions{:}}
% bbci.train_file = {[bbci.subdir bbci.fileNames{2} '*'], [bbci.subdir bbci.fileNames{3} '*']}
% bbci.train_file = {[bbci.subdir bbci.fileNames{3} '_' CLSTAG '*'] , [bbci.subdir bbci.fileNames{3} '_' CLSTAG '*'] };%,  fnamePreviousSessions{:}}
% bbci.train_file = {[bbci.subdir bbci.fileNames{2} '_' CLSTAG '*'], [bbci.subdir fname_auditory_MI]};

% bbci.train_file = {[bbci.subdir 'imag_fb_bar_pcovmean_getFour_VPlh_FR'], fnamePreviousSessions{:}};

% bbci.train_file = {fnamePreviousSessions{:}};
% bbci.train_file= {[bbci.subdir 'imag_fb_bar_pcovmean_getFour_VPlh_FR']}

bbci.save_name= [TODAY_DIR bbci.classyNames{1} '_' CLSTAG];

%  the std setup_opts are already set in     bbci_setup_csp_LRP
%  uncomment the field for manual editing!

%     bbci.setup_opts.model= {{'RLDAshrink', 'scaling',1, 'store_means',1, 'store_invcov',1,'store_extinvcov',1} {'RLDAshrink'} };
%     bbci.setup_opts.clab= {'not', 'Fp*'};
%     bbci.setup_opts.band = [9 12];
%     bbci.setup_opts.ival = [1000 4000];
%     bbci.setup_opts.do_laplace = 0;
%     bbci.setup_opts.visu_laplace = 0;

bbci.adaptation.running= 1; 
bbci.adaptation.policy = 'pcovmean_multi';
%extract the marker corresponding to the classes (foot-foot). bbci.classDef
% has all infos, bbci.classes just the labels to use...


bbci_bet_prepare
bbci.adaptation.mrk_end = [11 12 13 101:103 111:113];

bbci.adaptation.load_tmp_classifier =  0;
bbci.adaptation.UC_mean = 0.04;
bbci.adaptation.UC_pcov = 0.02;
bbci.adaptation.verbose = 2;
bbci.adaptation.ix_adaptive_cls = [1 2];
bbci.adaptation.adaptation_ival = bbci.setup_opts.ival; %?!?
bbci.setup_opts.do_laplace = 0;

% bbci.setup_opts.ival = {[1500 6000] [3000 4800]};
% bbci.setup_opts.band = {[20 24] [.1 6]};
    bbci.setup_opts.ival = {[3600 5000] [500 1200]};
%         bbci.setup_opts.ival = {[1000 2500] [450 1000]};
    bbci.setup_opts.band = {[15 24], [0.1 3]};%CSP/LRP (LRP is not really used ?!?)
    

bbci.setup_opts.visu_ival = [-1000 8000];
bbci.setup_opts.visu_band = [0.1 45];
bbci.setup_opts.do_LRP = 1;
bbci.setup_opts.reject_opts = {'do_multipass' , 1 , 'whiskerperc' , 10, 'whiskerlength', 1.8 , 'do_relvar' , 1 , 'do_bandpass', 0};


bbci_bet_analyze


dum = bbci.classDef(1,[find(strcmp(bbci.classDef(2,:), bbci.classes(1))) find(strcmp(bbci.classDef(2,:), bbci.classes(2)))] );
%     keyboard
bbci.adaptation.mrk_start = bbci.classDef(1,:);
if length(bbci.adaptation.mrk_start) ~= 2
    warning('there should be a mrk_start for each class for adaptation!!!')
end

fprintf('Type ''bbci_bet_finish'' to save classifier and ''dbcont'' proceed.\n');
keyboard
close all
bbci_bet_finish %% only if you are really sure about that!



%% Run1b
%% start classifier of last session and set bias/gradient

cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d]; ', VP_CODE, TODAY_DIR, VP_SCREEN);
cmd_bbci= ['dbstop if error ; bbci_bet_apply(''' bbci.save_name ''');'];
%cmd_bbci= ['dbstop if error ; bbci_bet_apply(''' classyNamePreviousSession ''');']; 
system(['matlab -nosplash -r "' cmd_init 'setup_tobi_games_MI; ' cmd_bbci '; exit &']);

%% calibration with online (bar) feedback !
nRuns= 1;
for ri= 1:nRuns,
    sprintf('\n\n');disp(['Start the software in mode '''  CLSTAG ''' and set showBars = TRUE'])
    inp = '';
    while ~strcmp(inp, 'start')
        sprintf('\n')
        inp = input('type ''start'' if you want to start the recording: ', 's');
    end


    fname = [bbci.fileNames{3} '_' CLSTAG];
    signalServer_startrecoding(fname)
    sprintf('\n\n');disp(['NO FEEDBACK RUN:    please start the calibration in the game, mode ''' CLSTAG ''' '])
    inp = '';
    while ~strcmp(inp, 'next')
        sprintf('\n')
        inp = input('type ''next  if you want to continue to the next run: ', 's');
    end
    ppTrigger(254) % just to ensure that the recording is stopped!
end

 input('stop the classifier!');

%% Train classifier on the previous Runs (TODAY) plus the old sessions
bbci.train_file = {[bbci.subdir bbci.fileNames{2} '_' CLSTAG '*'], ...
    [bbci.subdir bbci.fileNames{3} '_' CLSTAG '*'], ...  
}; %,  fnamePreviousSessions{:}

%   bbci.train_file = {[bbci.subdir bbci.fileNames{2} '_' CLSTAG '*'], ...
%       fnamePreviousSessions{:}};
%   
%   bbci.train_file = {[bbci.subdir 'imag*'], fnamePreviousSessions{:}}
% bbci.train_file = {[bbci.subdir bbci.fileNames{3} '_' CLSTAG '*'], [bbci.subdir bbci.fileNames{2} '_' CLSTAG '*']}%, ...  


% bbci.train_file = {[bbci.subdir 'imag_fb_bar_pcovmean_getFour_VPlh_FR'], fnamePreviousSessions{:}};

bbci.save_name= [TODAY_DIR bbci.classyNames{2} '_' CLSTAG];


bbci.adaptation.running= 0; 
bbci.adaptation.policy = 'pcovmean_multi'; %only works for cls(1)

%for Klaus, Moday night...
bbci.adaptation.running= 1; 
bbci.adaptation.policy = 'pcovmean_multi'; %only works for cls(1)
bbci.save_name= [TODAY_DIR bbci.classyNames{1} '_' CLSTAG];

bbci_bet_prepare
bbci.adaptation.mrk_end = [11 12 13 101:103 111:113];

bbci.adaptation.load_tmp_classifier =  0;
bbci.adaptation.UC_mean = 0.035;
bbci.adaptation.UC_pcov = 0.02;
bbci.adaptation.verbose = 2;
bbci.adaptation.verbose = 2;
bbci.adaptation.ix_adaptive_cls = [1 2];
% bbci.adaptation.ix_adaptive_cls = [2]; %only LRP!!
bbci.adaptation.adaptation_ival = bbci.setup_opts.ival; %?!?
bbci.setup_opts.do_laplace = 0;
bbci.setup_opts.ival = {[1500 4000], [2500 4000]};
bbci.setup_opts.band = {[10 13] [0.1 6]};
bbci.setup_opts.visu_ival = [-500 11000];%CSP
bbci.setup_opts.reject_opts = {'do_multipass' , 1 , 'whiskerperc' , 10, 'whiskerlength', 1.5 , 'do_relvar' , 1 , 'do_bandpass', 0};
bbci_bet_analyze

dum = bbci.classDef(1,[find(strcmp(bbci.classDef(2,:), bbci.classes(1))) find(strcmp(bbci.classDef(2,:), bbci.classes(2)))] );
%     keyboard
bbci.adaptation.mrk_start = [1 5];
if length(bbci.adaptation.mrk_start) ~= 2
    warning('there should be a mrk_start for each class for adaptation!!!')
end

fprintf('Type ''bbci_bet_finish'' to save classifier and ''dbcont'' proceed.\n');
keyboard
close all
bbci_bet_finish %% only if you are really sure about that!
 
%% - Runs 2
%% - BBCI adaptive Feedback (pretrained classifier), pcovmean adaptation
%desc= stimutil_readDescription('season11_imag_fbarrow_LRF_1');
%stimutil_showDescription(desc, 'clf',1, 'waitfor',0);
%stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig', 'waitfor_msg', 'Press <RETURN> to view the remaining instructions: ');
%desc= stimutil_readDescription('season11_imag_fbarrow_LRF_2');
%stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig', 'waitfor_msg', 'Press <RETURN> to start feedback: ');

nRuns= 2;
% setup_file= 'season11\cursor_adapt_pcovmean.setup';
% setup= nogui_load_setup(setup_file); %OK, take the patients stuff!
% today_vec= clock; today_str= sprintf('%02d_%02d_%02d', today_vec(1)-2000, today_vec(2:3));
% tmpfile= [TODAY_DIR 'adaptation_pcovmean_Lap_' today_str];
% % tag_list= {'LR', 'LF', 'FR'};
% tag_list= {'LR'}; %... to make it simpler, only LR class
% all_classes= {'left', 'right', 'foot'};
% warning('only training on two classes');

% % % % % % % % %
tag_list = {'FR'};

for ri= 1:nRuns,
    for ti= 1:length(tag_list),
        CLSTAG= tag_list{ti};
        cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d]; ', VP_CODE, TODAY_DIR, VP_SCREEN);
        bbci_cfy= [TODAY_DIR bbci.classyNames{2} '_' CLSTAG];
        cmd_bbci= ['dbstop if error ; bbci_bet_apply(''' bbci_cfy ''');'];
        system(['matlab -nosplash -r "' cmd_init 'setup_tobi_games_MI; ' cmd_bbci '; exit &']);
        %system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']);
        pause(6)

        fprintf('Starting Run %d %d: classes %s.\n', ceil(ri/2), ti+3*mod(ri-1,2), CLSTAG);
        fname = [bbci.fileNames{2} '_' CLSTAG];
        signalServer_startrecoding(fname)

        disp(['please start the calibration in the game, mode ' CLSTAG])
        %   %keyboard; fprintf('Thank you for letting me know ...\n');
        inp = '';
        while ~strcmp(inp, 'next')
            inp = input('type ''next  if you want to continue to the next run', 's');
        end
        ppTrigger(254) % just ensure that the recording was stopped!
    end
end


%% - Train 'CSP_16chans' on Feedbacks Runs
bbci= bbci_default;

% CLSTAG = input('Please specify the Classes (CLSTAG): options {LR, LF, RF}', 's');

bbci.train_file= {[bbci.subdir bbci.fileNames{1} '_' CLSTAG '*'], ...
     [bbci.subdir bbci.fileNames{2} '_' CLSTAG '*']};

bbci.save_name= [TODAY_DIR, bbci.classyNames{2} '_' CLSTAG];
bbci.adaptation.running= 1; 
bbci.adaptation.policy = 'pmean_multi';
bbci.adaptation.mrk_start = {1, 2, 3};
bbci.adaptation.load_tmp_classifier =  0;
bbci.adaptation.UC_mean = 0.075;
bbci.adaptation.UC_pcov = 0.03;
bbci.adaptation.verbose = 1;
bbci.setup_opts.do_laplace = 0;

%     bbci.setup_opts.iopt.ival = {[1200 2000], 'auto'};

bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all
bbci.adaptation.running= 1;
bbci.adaptation.policy = 'pmean';
bbci_bet_finish


%% - FAIR MODE
% bbci= bbci_default;
 CLSTAG = 'LR';%input('Please specify the Classes (CLSTAG): options {LR, LF, RF}', 's');


fname = [bbci.fileNames{4} '_' CLSTAG];

%start classifier
% class_init = ['send_tobi_c_udp(''init'', bbci.fb_machine, bbci.fb_port, {' ...
%     '{''ERD'' ''BBCI Motor Imagery ERD Classifier'', ''dist'', ''biosig'', ''0x300''},' ...
%     '{''LRP'' ''BBCI Motor Imagery LRP Classifier'', ''dist'', ''biosig'', ''0x300''},' ...
%     '{''fuse'' ''BBCI Motor Imagery fused Classifier'', ''dist'', ''biosig'', ''0x300''}' ...
%     '})'];
% cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d]; ', VP_CODE, TODAY_DIR, VP_SCREEN);
% bbci_cfy= [TODAY_DIR bbci.classyNames{2} '_' CLSTAG];
% cmd_bbci= ['dbstop if error ; bbci_bet_apply(''' bbci_cfy ''');'];
% system(['matlab -nosplash -r "' cmd_init 'setup_tobi_games_MI; ' cmd_bbci '; exit &']);
% pause(10)

signalServer_startrecoding(fname)
disp(['please start playing the game in mode ' CLSTAG])

inp = '';
while ~strcmp(inp, 'cancel')
    sprintf('\n')
    if strcmp(inp, 'r')
        ppTrigger(198)
    elseif strcmp(inp, 'f')
                ppTrigger(199)
    elseif strcmp(inp, '0')
                ppTrigger(197)
    elseif strcmp(inp, 'q') %LINKS gewollt
                ppTrigger(191)
    elseif strcmp(inp, 'w') %rechts gewollt
                ppTrigger(192)
    end
    inp = input('type ''r'' ''f'' ''0'' or ''cancel'': ', 's');
end
ppTrigger(254) % just to ensure that the recording is stopped!