addpath([BCI_DIR 'acquisition/setups/season10']);
addpath([BCI_DIR 'acquisition/setups/RSVP']);
addpath([BCI_DIR 'acquisition/setups/RSVP/online']);
addpath([BCI_DIR 'acquisition/setups/RSVP/toolbox_overloader']);
fprintf('\n\nWelcome to RSVP *** ONLINE ***\n\n');

%% Start BrainVisionn Recorder, load workspace and check triggers
system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');
bvr_sendcommand('loadworkspace', 'ActiCap_64ch_linked_mastoids.rwksp');
try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end


%% Check VP_CODE, initialize counter, and create data folder
if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end
if strcmp(VP_CODE, 'Temp');
  VP_NUMBER= 1;
else
  VP_COUNTER_FILE= [DATA_DIR 'RSVP_online_VP_Counter'];
  % delete([VP_COUNTER_FILE '.mat']);   %% for reset
  if exist([VP_COUNTER_FILE '.mat']),
    load(VP_COUNTER_FILE, 'VP_NUMBER');
  else
    VP_NUMBER= 0;
  end
  VP_NUMBER= VP_NUMBER + 1;
  fprintf('VP number %d.\n', VP_NUMBER);
end
global TODAY_DIR
acq_makeDataFolder('multiple_folders',1);

%% Settings for online classification
general_port_fields= struct('bvmachine','127.0.0.1',...
                            'control',{{'127.0.0.1',12471,12487}},...
                            'graphic',{{'',12487}});
general_port_fields.feedback_receiver= 'pyff';
bbci= [];
bbci.setup= 'ERP_Speller';
bbci.feedback= 'ERP_Speller';
bbci.clab= {'not','E*','Fp*','AF3,4','F9,10','FT*','I*'};
bbci.classDef= {[71:100], [31:60];
                'target', 'nontarget'};
bbci.marker_output.marker= num2cell([71:100, 31:60]);
bbci.marker_output.value= [1:30, 1:30];
% TO DO -> marker start of each countdown (before each trial) 
%  --->TRIG_COUNTDOWN_START = 200 ,TRIG_COUNTDOWN_END = 201
bbci.start_marker= [242];  %% Fixation start
% What happens when complete copyspelling is finished (is a marker sent??)
% @Benjamin: How again can we have bbci_bet_apply stop when pressing a
% button (eg 'R  1')
% question : should we delete/desable the triggers burst start/burst end? Maybe
% relpace them with sequence start/end and trial start/end?
bbci.quit_marker= [2 4 8 255];
bbci.pause_after_quit_marker= 1000;
bbci_default = bbci;

VP_SCREEN = [0 0 1920 1200];
fprintf('Type ''run_online_RSVP'' and press <RET>\n');
