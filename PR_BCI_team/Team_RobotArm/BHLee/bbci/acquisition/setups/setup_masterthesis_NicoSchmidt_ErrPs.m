if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

addpath([BCI_DIR 'acquisition/setups/masterthesis_NicoSchmidt_ErrPs']);

fprintf('\n\nWelcome to visual ErrP \n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

% Load Workspace into the BrainVision Recorder
% bvr_sendcommand('loadworkspace', 'ActiCap_64ch_EOGvu'); % used for ErrP calibration study
bvr_sendcommand('loadworkspace', 'ActiCap_64ch_EOGvu'); 

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR
acq_makeDataFolder('multiple_folders', 1);
%mkdir([TODAY_DIR 'data']);
VP_SCREEN = [0 0 1920 1180];
% VP_SCREEN = [0 0 1280 1024]; % eyetracker Bildschirm


%% Online settings
general_port_fields.feedback_receiver= 'pyff';
COPYSPELLING_FINISHED = 246;
bbci= [];
bbci.setup= 'ERP_Speller_ErrP_detection';
bbci.feedback= 'ERP_Speller_ErrP_detection';
bbci.func_mrk = 'mrkodef_ERP_Speller';
func_opts = struct();
func_opts.miscDef = {252, 253, 240, 244, 245, 99, 90, 91;
                   'run_start', 'run_end', 'countdown_start', ...
                   'end_level1', 'end_level2', 'invalid', 'machine_error', 'user_error'};
func_opts.ErrPDef = {96; 'ErrP'};
func_opts.nRepetitions = 10; 
bbci.func_mrk_opts = func_opts;
clear func_opts;
bbci.classDef= {[31:36 41:49], [11:16 21:29], [51:56 61:66 151:156 161:166]; ...
                  'target',     'nontarget',          'feedback'};
bbci.start_marker= [252];
bbci.quit_marker= [4 5 COPYSPELLING_FINISHED];
bbci.pause_after_quit_marker = 0;
bbci.withgraphics = 1;
bbci_default = bbci;
