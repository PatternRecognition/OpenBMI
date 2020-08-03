addpath([BCI_DIR 'acquisition/setups/leitstand2011_sma']);
fprintf('\n\nWelcome to Leitstand 2011 - SmA Online.\n\n');
  
%% Start BrainVisionn Recorder, load workspace and check triggers
system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');
bvr_sendcommand('loadworkspace', 'leitstand2011_sma');
try
  bvr_checkparport('type','R');
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
bbci.clab= {'not','EDA','Puls','Resp'};
bbci.setup= 'ERP_Speller';
bbci.feedback= '';
bbci.fb_machine= '192.168.1.23';
bbci.classDef= {[1 2 3 11 21];
                'message'};
bbci.nclassesrange= [1 1];
bbci.start_marker= 40;
bbci.quit_marker= 255;
bbci.pause_after_quit_marker= 1000;
bbci_default = bbci;

fprintf('Type ''run_leitstand2011_sma'' and press <RET>\n');
