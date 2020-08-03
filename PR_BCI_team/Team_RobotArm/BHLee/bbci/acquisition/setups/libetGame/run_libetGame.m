%% Recording - Calibration
for run= 1,
  fprintf('Press <RETURN> to start calibation run %d.\n', run), pause;
  bvr_startrecording(['calibration_LibetGame_' VP_CODE]);
%  pause(7.5*60);
  monitor_selfpaced('nTriggers',150);
  bvr_sendcommand('stoprecording');
end
  
%% Train the classifier
bbci= bbci_default;
bbci.train_file= strcat(TODAY_DIR, 'calibration_LibetGame_', VP_CODE, '02');
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_LibetGame_', VP_CODE, '02');

bbci_bet_prepare
mrk_tap= mrk;
mrk_pretap= mrk;
mrk_pretap.pos= mrk_pretap.pos - 2500/1000*mrk_pretap.fs;
mrk_pretap.className= {'rest'};
mrk= mrk_mergeMarkers(mrk_tap, mrk_pretap);
mrk = mrk_sortChronologically(mrk);
bbci_bet_analyze
fprintf('Type ''dbcont'' to continue\n');
keyboard
bbci_bet_finish
close all
    
%% Online LibetGame
pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'], 'bvplugin',0, 'gui',0);
send_xmlcmd_udp('init', '127.0.0.1', 12345);

startup_new_bbci_online

cfy_name= bbci.save_name;
bbci = bbci_apply_loadSettings(cfy_name);
global general_port_fields
bbci.source.acquire_fcn= @bbci_acquire_generic;
bbci.source.acquire_param= {100, ...
                           general_port_fields.bvmachine};
bbci.source.min_blocklength= 10;
bbci.control.fcn= @bbci_control_LibetGame;
bbci.feedback.receiver = 'pyff';
bbci.log.output= 'screen&file';
bbci.log.filebase= ['$TODAY_DIR' filesep 'log_LibetGame' VP_CODE];
bbci.log.classifier= 1;

configfile= 'C:\Vision\Recorder\BrainAmp.config';
copyfile([configfile '_10ms'], configfile);

for run= 1:5,
  pyff('init', 'LibetLight'); 
  fprintf('Press <RETURN> to start LibetGame, run %d.\n', run), pause;
  pyff('play', 'basename','online_LibetGame', 'impedances',0);
  data= bbci_apply(setfield(bbci,'quit_condition',struct('running_time',5*60)));
  fprintf('Game finished.\n')
  pyff('quit');
end

copyfile([configfile '_default'], configfile);


return
%% Restart

startup_new_bbci_online

bbci.save_name= strcat('D:/temp/bbci_classifier_LibetGame_', VP_CODE);

bbci = bbci_apply_loadSettings(bbci.save_name);
global general_port_fields
bbci.source.acquire_fcn= @bbci_acquire_generic;
bbci.source.acquire_param= {100, ...
                           general_port_fields.bvmachine};
bbci.source.min_blocklength= 10;
bbci.control.fcn= @bbci_control_LibetGame;
bbci.feedback.receiver = 'pyff';
bbci.log.output= 'screen&file';
bbci.log.filebase= ['$TODAY_DIR' filesep 'log_LibetGame' VP_CODE];
bbci.log.classifier= 1;

pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'],'gui',0, 'bvplugin',0);
pause(7);
send_xmlcmd_udp('init', '127.0.0.1', 12345);
pyff('init', 'LibetLight');  % Name des Feedbacks einf�gen
pause(1);

data= bbci_apply(setfield(bbci,'quit_condition',struct('running_time',1)));

pyff('play');
data= bbci_apply(bbci);
