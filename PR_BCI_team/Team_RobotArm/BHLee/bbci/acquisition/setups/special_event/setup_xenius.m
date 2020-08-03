addpath([BCI_DIR 'acquisition/setups/special_event/xenius_2010_03_17']);
fprintf('\n\nWelcome to the x:enius experiment.\n\n');
  
%% Start BrainVisionn Recorder, load workspace and check triggers
system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');
bvr_sendcommand('loadworkspace', 'ActiCap_64ch_linked_mastoids');
try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
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
bbci.feedback= 'ERP_Speller';
bbci.classDef= {[31:49], [11:29];
                'target', 'nontarget'};
bbci.start_marker= [240];           %% Countdown start
bbci.quit_marker= [2 246 255];  %% Push button or end markers
bbci.pause_after_quit_marker= 1000;
bbci_default = bbci;

%VP_SCREEN = [0 0 1920 1200];
VP_SCREEN = [-1280 0 1280 1024];
fprintf('Type ''run_xenius'' and press <RET>\n');
