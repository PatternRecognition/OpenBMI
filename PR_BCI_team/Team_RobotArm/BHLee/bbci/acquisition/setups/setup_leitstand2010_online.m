addpath([BCI_DIR 'acquisition/setups/leitstand2010_online']);
fprintf('\n\nWelcome to Leitstand 2010 - Online.\n\n');
  
%% Start BrainVisionn Recorder, load workspace and check triggers
system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');
bvr_sendcommand('loadworkspace', 'ActiCap_64ch_linked_mastoids');
try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

%% Check VP_CODE, initialize counter, and create data folder
if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end
global TODAY_DIR
acq_makeDataFolder;

%% Settings for online classification
general_port_fields= struct('bvmachine','127.0.0.1',...
                            'control',{{'127.0.0.1',12471,12487}},...
                            'graphic',{{'',12487}});
general_port_fields.feedback_receiver= 'pyff';
bbci= [];
bbci.setup= 'ERP_Speller';
bbci.feedback= '';
bbci.classDef= {[21 22];
                'message'};
bbci.nclassesrange= [1 1];
bbci.start_marker= [240];           %% Countdown start
bbci.quit_marker= [2 4 8 253 255];  %% Push button or end markers
bbci.pause_after_quit_marker= 1000;
bbci_default = bbci;

%VP_SCREEN = [0 10 1920 1180];
VP_SCREEN = [0 0 1280 1024];
fprintf('Type ''run_leitstand2010_online'' and press <RET>\n');
