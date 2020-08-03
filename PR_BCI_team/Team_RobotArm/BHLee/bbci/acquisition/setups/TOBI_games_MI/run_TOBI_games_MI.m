%% TODOs
% learn subject specific classifier!
% learn indendent 16ch cls_ERD_16chan_TOBIgames

% classy: -
% Datenaufnahme ca. 30/30:
% fileNames{1}= imag_calib_bar_getFour_VPxx01
%
% train auf {1}: bbci_classifier_cspauto_16chans_pcovmean_VPxx
%
%
% classyNames{1}: bbci_classifier_cspauto_16chans_pcovmean
% Datenaufnahme ca. 30/30:
% fileNames{2}= imag_fb_bar_pcovmean_getFour_VPxx
%
% train auf {1,2}: bbci_classifier_cspauto_16chans_pmean_VPxx
% fileNames{3}= imag_fb_bar_pmean_getFour  (FAIR zu benutzen!)
%

%% IMPEDANCE

signalserver_impedance('server_config_MI_BUK.xml')


%% INITILIZATION
% copyfile('D:\data\bbciRaw\VPlh_11_07_29\bbci_classifier_csp_LRP_16chans_pcovmean_VPlh_FR.mat', TODAY_DIR)

input('please close the impedance software')

start_signalserver('server_config_MI_BUK.xml')

input('please DEBLOCK the parallel port!');
input('please CHECK if the SignalServer is running properly!');


% game = input('Please specify the game you want to play: ''getFour'' or ''jetris'' :\n    ', 's');
% switch game
%     case 'getFour'
        bbci.classDef = {[1], [2], [3]; 'left', 'right', 'foot'};
%     case 'jetris'
%         bbci.classDef = {[1], [2], [3]; 'Left', 'Right', 'Foot'};
%     otherwise
%         error('game not specified correctly. Only getFour and jetris are implemented!')
% end


bbci.GAME = 'getFour';
all_classes= {'left', 'right', 'foot'};

%% Record Impedances
% To be done via SignalAcquisitionServer

%% Standard Recordings: Augenbewegungen, Blinzeln, Entspannen mit Augen auf/zu

[seq, wav, opt]= setup_TOBI_games_MI_artifacts_demo('clstag', '');
fprintf('Press <RETURN> when ready to start ARTIFACT TEST measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt, 'test',1, 'useSignalServer', 1);


%% - Artefaktmessung: Augenbewegungen, Ruhemessung mit Augen auf und Augen zu
fprintf('\n\nArtifact recording.\n');
[seq, wav, opt]= setup_tobi_games_MI_artifacts('clstag', '');
fprintf('Press <RETURN> when ready to start ARTIFACT measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt,'filename', 'arte_amAnfang', 'useSignalServer', 1);



%% Run1
%% no feedback!
nRuns= 1;
% tag_list= {'LR', 'LF', 'FR'};
tag_list= {'LF'};
% tag_list= {'LF', 'FR'}
for ri= 1:nRuns,
    for ti= 1:length(tag_list),
        CLSTAG= tag_list{ti};
        fname = [bbci.fileNames{1} '_mentCalc' CLSTAG];
        signalServer_startrecoding(fname)
        sprintf('\n\n');disp(['NO FEEDBACK RUN:    please start the calibration in the game, mode ''' CLSTAG ''' '])
        inp = '';
        while ~strcmp(inp, 'next')
            sprintf('\n')
            inp = input('type ''next  if you want to continue to the next run: ', 's');
        end
        ppTrigger(254) % just to ensure that the recording is stopped!
    end
end


%% Train classifier on the No-feedback Runs
tag_list = {'LF'} %FOR TESTING
for ti= 1:length(tag_list),
    bbci= bbci_default;
    CLSTAG= tag_list{ti};
    bbci.train_file = [bbci.subdir bbci.fileNames{1} '_' CLSTAG '*'];

    bbci.save_name= [TODAY_DIR bbci.classyNames{1} '_' CLSTAG];

    %  the std setup_opts are already set in     bbci_setup_csp_LRP
    %  uncomment the field for manual editing!

    %     bbci.setup_opts.model= {{'RLDAshrink', 'scaling',1, 'store_means',1, 'store_invcov',1,'store_extinvcov',1} {'RLDAshrink'} };
    %     bbci.setup_opts.clab= {'not', 'Fp*'};
    %     bbci.setup_opts.band = [9 12];
    %     bbci.setup_opts.ival = [1000 4000];
    %     bbci.setup_opts.do_laplace = 0;
    %     bbci.setup_opts.visu_laplace = 0;
    
    bbci.adaptation.running= 1; %this only applies to cls(1) -->
    bbci.adaptation.policy = 'pcovmean_multi';
      %extract the marker corresponding to the classes (foot-foot). bbci.classDef
      % has all infos, bbci.classes just the labels to use...
    


    bbci_bet_prepare
    dum = bbci.classDef(1,[find(strcmp(bbci.classDef(2,:), bbci.classes(1))) find(strcmp(bbci.classDef(2,:), bbci.classes(2)))] );
    %     keyboard
    bbci.adaptation.mrk_start = dum;
    bbci.adaptation.mrk_end = [1 2 3 11 12 13];
    
    bbci.adaptation.load_tmp_classifier =  0;
    bbci.adaptation.UC_mean = 0.075;
    bbci.adaptation.UC_pcov = 0.03;
    bbci.adaptation.verbose = 2;
    bbci.adaptation.ix_adaptive_cls = [1 2];
    bbci.adaptation.adaptation_ival = bbci.setup_opts.ival;
    bbci.setup_opts.do_laplace = 0;
    bbci_bet_analyze


    fprintf('Type ''bbci_bet_finish'' to save classifier and ''dbcont'' proceed.\n');
    keyboard
    close all

    %bbci_bet_finish %% only if you are really sure about that!


end


%% - Runs 2
%% - BBCI adaptive Feedback (pretrained classifier), pcovmean adaptation
%desc= stimutil_readDescription('season11_imag_fbarrow_LRF_1');
%stimutil_showDescription(desc, 'clf',1, 'waitfor',0);
%stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig', 'waitfor_msg', 'Press <RETURN> to view the remaining instructions: ');
%desc= stimutil_readDescription('season11_imag_fbarrow_LRF_2');
%stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig', 'waitfor_msg', 'Press <RETURN> to start feedback: ');

nRuns= 1;
% setup_file= 'season11\cursor_adapt_pcovmean.setup';
% setup= nogui_load_setup(setup_file); %OK, take the patients stuff!
% today_vec= clock; today_str= sprintf('%02d_%02d_%02d', today_vec(1)-2000, today_vec(2:3));
% tmpfile= [TODAY_DIR 'adaptation_pcovmean_Lap_' today_str];
% % tag_list= {'LR', 'LF', 'FR'};
% tag_list= {'LR'}; %... to make it simpler, only LR class
% all_classes= {'left', 'right', 'foot'};
% warning('only training on two classes');

% % % % % % % % %
tag_list = {'LF'};

for ri= 1:nRuns,
    for ti= 1:length(tag_list),
        CLSTAG= tag_list{ti};
        cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d]; ', VP_CODE, TODAY_DIR, VP_SCREEN);
        bbci_cfy= [TODAY_DIR 'INITIAL11_' bbci.classyNames{1} '_mentCalc' CLSTAG];
        cmd_bbci= ['dbstop if error ; bbci_bet_apply(''' bbci_cfy ''');'];
        system(['matlab -nosplash -r "' cmd_init 'setup_tobi_games_MI; ' cmd_bbci '; exit &']);
        %system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']);
        pause(6)

        fprintf('Starting Run %d %d: classes %s.\n', ceil(ri/2), ti+3*mod(ri-1,2), CLSTAG);
        fname = [bbci.fileNames{2} '_mentCalc' CLSTAG];
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

bbci.train_file= {[bbci.subdir bbci.fileNames{1} '_' CLSTAG '*']}, ...
   % [bbci.subdir bbci.fileNames{2} '_' CLSTAG '*']};

bbci.save_name= [TODAY_DIR, bbci.classyNames{2} '_' CLSTAG];
bbci.adaptation.running= 1; %this only applies to cls(1) -->
    bbci.adaptation.policy = 'pmean';
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
bbci= bbci_default;
CLSTAG = input('Please specify the Classes (CLSTAG): options {LR, LF, RF}', 's');

EEG_fname = [bbci.subdir bbci.fileNames{3} '_' CLSTAG '*'];

%start classifier
% class_init = ['send_tobi_c_udp(''init'', bbci.fb_machine, bbci.fb_port, {' ...
%     '{''ERD'' ''BBCI Motor Imagery ERD Classifier'', ''dist'', ''biosig'', ''0x300''},' ...
%     '{''LRP'' ''BBCI Motor Imagery LRP Classifier'', ''dist'', ''biosig'', ''0x300''},' ...
%     '{''fuse'' ''BBCI Motor Imagery fused Classifier'', ''dist'', ''biosig'', ''0x300''}' ...
%     '})'];
cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d]; ', VP_CODE, TODAY_DIR, VP_SCREEN);
bbci_cfy= [TODAY_DIR bbci.classyNames{2} '_' CLSTAG];
cmd_bbci= ['dbstop if error ; bbci_bet_apply(''' bbci_cfy ''');'];
system(['matlab -nosplash -r "' cmd_init 'setup_tobi_games_MI; ' cmd_bbci '; exit &']);
pause(10)

signalServer_startrecoding(fname)
disp(['please start playing the game in mode ' CLSTAG])

inp = '';
while ~strcmp(inp, 'cancel')
    sprintf('\n')
    inp = input('type ''cancel'' if you want to CANCEL: ', 's');
end
ppTrigger(254) % just to ensure that the recording is stopped!