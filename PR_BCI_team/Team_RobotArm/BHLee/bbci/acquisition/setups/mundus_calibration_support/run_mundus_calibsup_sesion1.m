  global RAW_ETH;

fprintf('Welcome to session 1 of calibration support experiment!\n');

bvr_sendcommand('checkimpedances');
fprintf('Prepare cap. Press <RETURN> when finished.\n');
pause


%% TEST FES
RAW_ETH = 1; 
[stim, opt] = setup_mundus_calibsup_image_arrow({'left','down'}, 5, 2000, 3000);

fprintf('TEST: Press <RETURN> when ready to start ''nmes'' test 1.\n');
pause

stim_visualCuesExt(stim, opt, 'test',1, 'test_mode', 1, 'fes', 1);

%% TEST MI

RAW_ETH = 0; 
[stim, opt] = setup_mundus_calibsup_image_arrow({'left','right','down'}, 10, 2000, 3000);

fprintf('TEST: Press <RETURN> when ready to start ''imagined movements'' test.\n');
pause
stim_visualCuesExt(stim, opt, 'test',1, 'test_mode', 1);


%% PART 1
%--------------------------------------------------------------------------

%% FES RUN

RAW_ETH = 1;
[stim, opt] = setup_mundus_calibsup_image_arrow({'left','right','down'}, 25, 2000, 3000);
opt.filename = 'FES_ONLY';

for i=1:3
   fprintf('Press <RETURN> when ready to start ''electrical stimulation'' measurement.\n');
   pause
   stim_visualCuesExt(stim, opt, 'fes', 1);
end;
fprintf('end of only FES runs.\n');

%% MI RUN

RAW_ETH = 0;
[stim, opt] = setup_mundus_calibsup_image_arrow({'left','right','down'}, 25, 2000, 3000);
opt.filename = 'MI_ONLY';

for i=1:3
    fprintf('Press <RETURN> when ready to start ''imagined movement'' measurement.\n');
    pause
    stim_visualCuesExt(stim, opt);
end;
fprintf('end of MIruns.\n'); 


%% MI + FES RUN

RAW_ETH = 1;
[stim, opt] = setup_mundus_calibsup_image_arrow({'left','right','down'}, 25, 2000, 3000);
opt.filename = 'MI_FES';

for i=1:3
   fprintf('Press <RETURN> when ready to start ''imagined movement + electrical stimulation'' measurement.\n');
   pause
   stim_visualCuesExt(stim, opt);
end;
fprintf('end of MI + FES runs.\n'); 


%% data analysis to select the best 2 classes.

bbci.setup_opts.model= {'RLDAshrink', 'gamma',0, 'scaling',1, 'store_means',1};
%bbci.setup_opts.clab= {'F3-4','FC5-6','CFC5-6','C5-6','CCP5-6','CP5-6','P5-6','PO3,z,4'};
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspauto_48chans');
bbci.setup='cspauto';
bbci.subdir = 'D:\data\bbciRaw\VPm7a_10_12_15';
bbci.train_file= strcat(bbci.subdir, '/MI_*');
%bbci.impedance_threshold = Inf; % for testing purposes!!!!!

bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all
bbci_bet_finish


%% PART 2
%--------------------------------------------------------------------------
RAW_ETH = 1;
[stim, opt] = setup_mundus_calibsup_image_arrow({'left', 'foot'}, 40, 2000, 3000);
opt.filename = 'MI_AND_FES_worstlimb';

for i=1:2
    fprintf('Press <RETURN> when ready to start ''imagined movement + electrical stimulation'' measurement.\n');
    pause
    stim_visualCuesExt(stim, opt);
end;

fprintf('end of MI + FES runs.\n'); 

opt.filename = 'MI_AND_FES_samelimb';
for i=1:2
   fprintf('Press <RETURN> when ready to start ''imagined movement + electrical stimulation'' measurement.\n');
   pause
   stim_visualCuesExt(stim, opt);
end;