
%-newblock
bvr_sendcommand('checkimpedances');
fprintf('\nPrepare cap. Press <RETURN> when finished.\n');
pause

%-newblock
%% - Runs 1, 2
%% - BBCI adaptive Feedback (subject-independent classifier, log-bp[8-15;16-35] at Lap C3,4), pcovmean adaptation
%desc= stimutil_readDescription('season10_imag_fbarrow_LRF');
%stimutil_showDescription(desc, 'clf',1, 'waitfor',0);
%stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig', 'waitfor_msg', 'Press <RETURN> to start feedback: ');
%keyboard
nRuns= 3;
%setup_file= 'multitimescale\cursor_adapt_pcovmean.setup';
setup_file= 'season11\cursor_adapt_pcovmean.setup';
setup= nogui_load_setup(setup_file);
today_vec= clock;
today_str= sprintf('%02d_%02d_%02d', today_vec(1)-2000, today_vec(2:3));
tmpfile= [TODAY_DIR 'adaptation_pcovmean_Lap_' today_str];
tag_list= {'LR', 'LF', 'FR'};
CLSTAG= tag_list{1};
all_classes= {'left', 'right'};
  ci1= find(CLSTAG(1)=='LRF');
  ci2= find(CLSTAG(2)=='LRF');
  classes= all_classes([ci1 ci2]);
for ri= 1:nRuns,
 %for ti= 1 %:length(tag_list),  
  cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d];', VP_CODE, TODAY_DIR, VP_SCREEN);
  bbci_cfy= [TODAY_DIR '/Lap_C3z4_bp2_' CLSTAG];
  cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''');'];
 system(['matlab -nosplash -r "' cmd_init 'setup_pcovmean; ' cmd_bbci '; exit &']);
%   system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']);
  pause(7)
  
  settings_bbci= {'start_marker', 210, ...
                  'quit_marker', 254, ...
                  'feedback', '1d', ...
                  'fb_port', 12345, ...
                  'adaptation.policy', 'pcovmean', ...  % war: 'pcovmean test'
                  'adaptation.offset', 750, ...
                  'adaptation.tmpfile', [tmpfile '_' CLSTAG], ...
                  'adaptation.mrk_start', {ci1, ci2}, ...                  
                  'adaptation.UC_pcov', 0.03, ...
                  'adaptation.UC_mean', 0.075, ...
                  'adaptation.load_tmp_classifier', ri>1};
  settings_fb= {'classes', classes, ...
                'trigger_classes_list', {'left','right','foot'},  ...
                'pause_msg', [CLSTAG(1) ' vs. ' CLSTAG(2)], ...
                'trials_per_run', 100, ...
                'break_every', 20};
  setup.graphic_player1.feedback_opt= overwrite_fields(setup.graphic_player1.feedback_opt, settings_fb);
  setup.control_player1.bbci= overwrite_fields(setup.control_player1.bbci, settings_bbci);
  setup.general.savestring= ['imag_fbarrow_LapC3z4_' CLSTAG];
  pause(7);
  nogui_send_setup(setup);
%  fprintf('Press <RETURN> when ready to start Run %d°%d: classes %s and wait (press <RETURN only once!).\n', ri, ti, CLSTAG);
%  pause; fprintf('Ok, starting ...\n');
  fprintf('Starting Run %d: classes %s.\n', ceil(ri), CLSTAG);
  if ri>1,
    pause(5);  %% Give time to display class combination, e.g. 'L vs R'
  end
  nogui_start_feedback(setup, 'impedances',ri==1);
  fprintf('Press <RETURN> when feedback has finished (windows must have closed).\n');
  pause; fprintf('Thank you for letting me know ...\n');
end

