basename= 'leitstand2011sma';

%% Calibration leitstand - Phase 1: message display
stimutil_waitForInput('msg_next','to start CALIBRATION NO.1 recording.');
bvr_startrecording(['calibration_' basename VP_CODE]);
pause(20*60);
bvr_sendcommand('stoprecording');
fprintf('CALIBRATION No.1 is finished\n');
soundsc(randn(5000,1));

%% Calibration leitstand2 - Phase 2: with additional task
stimutil_waitForInput('msg_next','to start CALIBRATION NO.2 recording.');
bvr_startrecording(['calibration_' basename VP_CODE]);
pause(45*60);
bvr_sendcommand('stoprecording');
fprintf('CALIBRATION No.2 is finished\n');
soundsc(randn(5000,1));

%% Train the classifier
bbci= bbci_default;
bbci.train_file= strcat(TODAY_DIR, 'calibration_*');
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_leitstand_', VP_CODE);
bbci_bet_prepare
bbci.setup_opts.disp_ival= [-200 700];
bbci.setup_opts.cfy_clab= {'F3,z,4','FC3-4','C5-6','CP5-6','P#','PO#','O#','I#'};

% remove double triggers and too dense messages
dp= diff(mrk.pos/mrk.fs*1000);
idel= 1+find(dp<=2000);
mrk= mrk_chooseEvents(mrk, 'not',idel);
%% extract target markers during run 1:
mrk_r1= mrk_selectEvents(mrk, find(mrk.pos<=Cnt.T(1)));
%% extract target markers during primary task in run2:
blk_ls= blk_segmentsFromMarkers(mrk_orig, ...
    'start_marker','S 41','end_marker','S 40', ...
    'skip_unfinished',0);
blk_ls.ival(2,end)= sum(Cnt.T);
% correct for transition between tasks
blk_ls.ival(1,:)= blk_ls.ival(1,:) + 4*blk_ls.fs;
blk_ls.ival(2,:)= blk_ls.ival(2,:) - 1*blk_ls.fs;
mrk_r2= mrk_addBlockNumbers(mrk, blk_ls);

mrk_r2= mrk_selectEvents(mrk_r2, ~isnan(mrk_r2.block_no));

if strcmp(VP_CODE, 'VPlaa'),
  % in the first experiment with VPlaa, the second calibration
  % had messed up marker positions for messages
  mrk_targets= mrk_r1;
else
  mrk_targets= mrk_mergeMarkers(mrk_r1, mrk_r2);
end

%% extract nontarget markers from pre-message intervals:
mrk_premsg= mrk_setClasses(mrk_targets, 1:numel(mrk_targets.pos), 'rest');
mrk_premsg.pos= mrk_premsg.pos - bbci.setup_opts.disp_ival(2)/1000*mrk.fs;
%% extract nontarget markers during secondary task:
blk_cs= blk_segmentsFromMarkers(mrk_orig, ...
                                'start_marker','S 40','end_marker','S 41');
% correct for transition between tasks
blk_cs.ival(1,:)= blk_cs.ival(1,:) + 4*blk_cs.fs;
blk_cs.ival(2,:)= blk_cs.ival(2,:) - 1*blk_cs.fs;
mrk_cs= mrk_evenlyInBlocks(blk_cs, diff(bbci.setup_opts.disp_ival), 'offset_start',1000);
mrk_cs.y= ones(1, length(mrk_cs.pos));
mrk_cs.toe= mrk_cs.y;
mrk_cs.className= {'rest'};

mrk_nontargets= mrk_mergeMarkers(mrk_cs, mrk_premsg);
mrk= mrk_mergeMarkers(mrk_targets, mrk_nontargets);

mrk_t1= mrk_chooseEvents(mrk_r1, find(ismember(mrk_r1.toe,[1 2])));
mrk_t1.className= {'t-sprache'};
mrk_t2= mrk_chooseEvents(mrk_r1, find(ismember(mrk_r1.toe,[3 11 21])));
mrk_t2.className= {'t-still'};
mk= mrk_mergeMarkers(mrk_t1, mrk_t2, mrk_nontargets);

epo= cntToEpo(Cnt, mk, [-200 800]);
epo= proc_baseline(epo, [-200 0]);
grid_plot(epo, mnt, defopt_erps);
clear epo mk

%% train classifier
bbci_bet_analyze
fprintf('Type ''dbcont'' to continue\n');
keyboard
bbci_bet_finish
close all

%% Check impedances before starting online operation
bvr_sendcommand('checkimpedances');
stimutil_waitForInput('msg_next','when impedances are fine.');

%% Online leitstand
fprintf('Move display for entering tasks to the side.\n');
stimutil_waitForInput('msg_next','to start ONLINE recording.');
bvr_startrecording(['online_' basename VP_CODE]);
pause(15);
fprintf('Type ppTrigger(255) in a second Matlab after 35 min to stop.\n');
bbci_bet_apply(bbci.save_name, 'bbci.feedback','', 'bbci.fb_port', 12345);
% or stop with <Ctrl-C> after 35 min

%% 2
fprintf('Move display for entering tasks to the front.\n');
stimutil_waitForInput('msg_next','to start ONLINE recording.');
bvr_startrecording(['online_' basename VP_CODE]);
pause(15);
fprintf('Type ppTrigger(255) in a second Matlab after 35 min to stop.\n');
bbci_bet_apply(bbci.save_name, 'bbci.feedback','', 'bbci.fb_port', 12345);
% or stop with <Ctrl-C> after 35 min

%% 3
fprintf('Display for entering tasks to the side.\n');
stimutil_waitForInput('msg_next','to start ONLINE recording.');
bvr_startrecording(['online_' basename VP_CODE]);
pause(15);
fprintf('Type ppTrigger(255) in a second Matlab after 35 min to stop.\n');
bbci_bet_apply(bbci.save_name, 'bbci.feedback','', 'bbci.fb_port', 12345);
% or stop with <Ctrl-C> after 35 min

%% 4
fprintf('Display for entering tasks to the front.\n');
stimutil_waitForInput('msg_next','to start ONLINE recording.');
bvr_startrecording(['online_' basename VP_CODE]);
pause(15);
fprintf('Type ppTrigger(255) in a second Matlab after 35 min to stop.\n');
bbci_bet_apply(bbci.save_name, 'bbci.feedback','', 'bbci.fb_port', 12345);
% or stop with <Ctrl-C> after 35 min
