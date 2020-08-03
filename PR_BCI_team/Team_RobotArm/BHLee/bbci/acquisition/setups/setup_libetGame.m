addpath([BCI_DIR 'acquisition/setups/libetGame']);
fprintf('\n\nWelcome to the Libet Game.\n\n');
  
%% Start BrainVisionn Recorder, load workspace and check triggers
system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');
bvr_sendcommand('loadworkspace', 'reducerbox_64std_EMGf');
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
bbci.setup= 'LibetGame';
bbci.clab= {'not', 'E*'};
bbci.feedback= 'LibetGame';
bbci.classDef= {'R 13';
                'keypress'};
bbci.nclassesrange= [1 1];
bbci.quit_marker= [8 255];
bbci_default = bbci;

%VP_SCREEN = [0 0 1920 1200];
VP_SCREEN = [0 0 1280 1024];
fprintf('Type ''run_libetGame'' and press <RET>\n');
