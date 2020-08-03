% CharStreamer 
% @JohannesHoehne 

if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
  fprintf('\n\nPres Ctrl-C to stop')  
  pause(5)
end

% setup_bbci_online; %% needed for acquire_bv
startup_new_bbci_online;
addpath([BCI_DIR 'acquisition/setups/TOBI_students12_CharStreamer'])

global TODAY_DIR
acq_makeDataFolder('log_dir',1);

set_general_port_fields('localhost');
general_port_fields.feedback_receiver = 'pyff';

fprintf('\n\nWelcome to the students CharStreamer project \n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');


send_udp_xml('init', general_port_fields.bvmachine, 12345); %do we need that?!?

bvr_sendcommand('loadworkspace', 'reducerbox_64std');

pause(2)


fprintf('While steeing up the cap, it is time for the subject to get to know the stimuli !!\n\n So we play the Feedback by hand (PYFF GUI)! \n\n press <ENTER> to continue!');
pause(2)

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

bvr_sendcommand('viewsignals');

send_xmlcmd_udp('init', '127.0.0.1', 12345)






%% Define some variables
RUN_END= [254];
band = [.35 20];

clear bbci_default
bbci_default.source.acquire_fcn= @bbci_acquire_bv;
bbci_default.source.acquire_param= {struct('fs', 100)};
bbci_default.source.record_signals= 1;
bbci_default.source.record_basename= 'FreeSpelling_mode';
bbci_default.log.output= 'screen&file';
bbci_default.log.classifier= 1;
bbci_default.control.fcn= @bbci_control_ERP_Speller_binary;
bbci_default.feedback.receiver = 'pyff';
bbci_default.quit_condition.marker= RUN_END;

BC= [];
BC.folder= TODAY_DIR;


BC.read_param= {'fs',100}; 
BC.marker_fcn= @mrk_defineClasses;
BC.marker_param= {{[101:130], [1:30]; 'target', 'nontarget'}};
BC.save.file= 'bbci_classifier_ERP_Speller';
BC.fcn= @bbci_calibrate_ERP_Speller;
BC.settings= strukt(...
                  'disp_ival', [-500 1600], ...
                  'ref_ival', [], ... [-500 -350]
                  'band', band, ...
                  'cfy_clab', {'not','E*'}, ...
                  'cfy_ival', 'auto', ...
                  'cfy_ival_pick_peak', [-400 1600], ...
                  'control_per_stimulus', 1, ...
                  'model', 'RLDAshrink', ...
                  'nSequences', [], ...
                  'nClasses', 30, ...
                  'cue_markers', [1:30]);
BC.save.figures=1;


bbci= struct('calibrate', BC);

% eval(['run_' session_name '_script']);


fprintf('\n\n !! Please check whether PYFF was successfully set up!!! \n')


fprintf('Type ''run_CharStreamer_calibration'' and press <RET>.\n');
