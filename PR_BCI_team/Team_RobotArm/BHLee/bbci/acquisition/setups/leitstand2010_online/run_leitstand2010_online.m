pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'],'gui',1);

%% Practice
pyff('init', 'AlarmControl'); pause(2);
pyff('setdir','');
pyff('play');
stimutil_waitForMarker({'S253' 'S255' 'R  2' 'R  4' 'R  8'});
pyff('quit');
  
%% Calibration leitstand
fprintf('Press <RETURN> to start experiment.\n'); pause;
pyff('init', 'AlarmControl'); pause(2);
% specific settings for first training
pyff('setint', 'cs_duration', 30);  % duration of the copyspeller task
pyff('setint', 't_copyspeller', 0); % interval to schedule the cs task
pyff('set', 'can_stop_on_control_signal', false); % stop cs task if appropriate control signal was sent?

pyff('setdir', 'basename',['calibration_leitstand']);
pyff('save_settings', 'leitstand2010_online_calib1');
pyff('play');
stimutil_waitForMarker({'S253' 'S255' 'R  2' 'R  4' 'R  8'});
pyff('quit');

%% Calibration leitstand2
fprintf('Press <RETURN> to start experiment.\n'); pause;
pyff('init', 'AlarmControl'); pause(2);
% specific settings for second training
pyff('setint', 'cs_duration', 30);  % duration of the copyspeller task
pyff('setint', 't_copyspeller', 90); % interval to schedule the cs task
pyff('set', 'can_stop_on_control_signal', false); % stop cs task if appropriate control signal was sent?

pyff('setdir', 'basename',['calibration_leitstand']);
pyff('save_settings', 'leitstand2010_online_calib2');
pyff('play');
stimutil_waitForMarker({'S253' 'S255' 'R  2' 'R  4' 'R  8'});
pyff('quit');


%% Train the classifier
bbci= bbci_default;
bbci.train_file= strcat(TODAY_DIR, 'calibration_*');
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_leitstand_', VP_CODE);
bbci_bet_prepare
bbci.setup_opts.disp_ival= [-200 700];
bbci.setup_opts.cfy_clab= {'F3,z,4','FC3-4','C5-6','CP5-6','P#','PO#','O#'};

mrk_all= mrk;
%% extract target markers during run 1:
mrk_r1= mrk_selectEvents(mrk, find(mrk.pos<=Cnt.T(1)));
%% extract target markers during primary task in run2:
blk_ls= blk_segmentsFromMarkers(mrk_orig, ...
    'start_marker','S150','end_marker','S 50', ...
    'start_first_block', Cnt.T(1)+1, ...
    'skip_unfinished',0);
blk_ls.ival(2,end)= sum(Cnt.T);
mrk_r2= mrk_addBlockNumbers(mrk, blk_ls);
mrk_r2= mrk_selectEvents(mrk_r2, ~isnan(mrk_r2.block_no));

mrk_targets= mrk_mergeMarkers(mrk_r1, mrk_r2);

%% extract nontarget markers from pre-message intervals:
mrk_premsg= mrk_setClasses(mrk_targets, 1:numel(mrk_targets.pos), 'rest');
mrk_premsg.pos= mrk_premsg.pos - bbci.setup_opts.disp_ival(2)/1000*mrk.fs;
%% extract nontarget markers during secondary task:
blk_cs= blk_segmentsFromMarkers(mrk_orig, 'start_marker','S 80','end_marker','S180');
mrk_cs= mrk_evenlyInBlocks(blk_cs, diff(bbci.setup_opts.disp_ival), 'offset_start',1000);
mrk_cs.y= ones(1, length(mrk_cs.pos));
mrk_cs.toe= mrk_cs.y;
mrk_cs.className= {'rest'};

mrk_nontargets= mrk_mergeMarkers(mrk_cs, mrk_premsg);
mrk= mrk_mergeMarkers(mrk_targets, mrk_nontargets);

bbci_bet_analyze
fprintf('Type ''dbcont'' to continue\n');
keyboard
bbci_bet_finish
close all

%% Online leitstand
fprintf('Press <RETURN> to start the online part.\n'), pause;
pyff('init', 'AlarmControl'); pause(2);
% specific settings for second training
pyff('setint', 'cs_duration', 30);  % duration of the copyspeller task
pyff('setint', 't_copyspeller', 90); % interval to schedule the cs task
pyff('set', 'can_stop_on_control_signal', true); % stop cs task if appropriate control signal was sent?

pyff('setdir', 'basename',['online_leitstand']);
pyff('save_settings', 'leitstand2010_online_online1');
pyff('play');
pause(10);
bbci_bet_apply(bbci.save_name, 'bbci.feedback','', 'bbci.fb_port', 12345);
fprintf('Experiment finished.\n')
pyff('stop'); pyff('quit');
