fprintf('\n\nWelcome to Lovisas Labrotation.\n\n');
addpath([BCI_DIR 'acquisition/setups/labrotation2010_LovisaHelgadottir']);

if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end
if strcmp(VP_CODE, 'Temp');
  VP_NUMBER= 1;
else
  VP_COUNTER_FILE= [DATA_DIR 'labrotation2010_LovisaHelgadottir_VP_Counter'];
  % delete([VP_COUNTER_FILE '.mat']);   %% for reset
  if exist([VP_COUNTER_FILE '.mat']),
    load(VP_COUNTER_FILE, 'VP_NUMBER');
  else
    VP_NUMBER= 0;
  end
  VP_NUMBER= VP_NUMBER + 1;
  fprintf('VP number %d.\n', VP_NUMBER);
end

% Load Workspace into the BrainVision Recorder
system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');
bvr_sendcommand('loadworkspace', 'ActiCap_64ch_linked_mastoids');
%bvr_sendcommand('loadworkspace', 'ActiCap_64ch_EOGvu');

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR
acq_makeDataFolder;
%VP_SCREEN = [0 10 1920 1200];
%VP_SCREEN = [1680 0 1280 1024];  % bbcilaptop02
VP_SCREEN = [0 0 1280 1024];  % bbcilaptop05

%% Online settings
general_port_fields= struct('bvmachine','127.0.0.1',...
                            'control',{{'127.0.0.1',12471,12487}},...
                            'graphic',{{'',12487}});
general_port_fields.feedback_receiver= 'pyff';
bbci= [];
bbci.setup= 'ERP_Speller';
bbci.feedback= 'ERP_Speller';
bbci.classDef= {[31:49], [11:29];
                'target', 'nontarget'};
bbci.start_marker= [240];  %% Countdown start
bbci.quit_marker= [2 4 8 246 253 255];
bbci.pause_after_quit_marker= 1000;
bbci_default = bbci;

fprintf('Type ''run_labrotation2010_LovisaHelgadottir'' and press <RET>\n');
