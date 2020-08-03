general_port_fields= struct('bvmachine','127.0.0.1',...
                            'control',{{'127.0.0.1',12471,12487}},...
                            'graphic',{{'',12487}});
general_port_fields.feedback_receiver= 'pyff';

%% Relaxation
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_season10_relax;
opt = rmfield(opt,'handle_cross');
seq= ['P2000 F21 P3000 F14P300000 F1P5000 F15P300000 F1P5000 F20P1000'];

fprintf('Press <RETURN> to start RELAX measurement.\n');
pause; fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt);

%% Experimental design
RUN_END = 255;
nDirections = 3;
nTrialsPerDirection = 100;
nTrials = nDirections * nTrialsPerDirection;
nPracticeTrials = nDirections * 3;

instructionDir = [BCI_DIR 'acquisition/setups/project10_alpha'];
basename = 'alpha_';
VEP_file = [BCI_DIR 'acquisition/setups/project10_alpha/VEP_feedback'];

%% Pyff
pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks\_VisualSpeller'],'gui',0);

%% Vormessung: VEP
pyff('init','CheckerboardVEP');pause(1)
pyff('load_settings', VEP_file);
pyff('setint','screen_pos',VP_SCREEN);
fprintf('Press <RETURN> to start VEP measurement.\n'); pause;
fprintf('Ok, starting...\n'),close all
%pyff('setdir', '');
pyff('setdir', 'basename', 'VEP');
pause(5)
pyff('play')
pause(5)
stimutil_waitForMarker(RUN_END);
fprintf('VEP measurement finished.\n')
bvr_sendcommand('stoprecording');
pyff('stop');
pyff('quit');
fprintf('Press <RET> to continue.\n'); pause;

%% Practice
log_filename = [TODAY_DIR  'practice_' basename VP_CODE '.log']; %#ok<*NASGU>
desc= stimutil_readDescription('alpha_instruction_practice','dir',instructionDir);
h_desc= stimutil_showDescription(desc, 'waitfor',0,'clf',1);

setup_alpha_feedback;
pyff('setint','nr_trials',nPracticeTrials);
fprintf(['Press <RET> to start PRACTICE.\n']);
pause, fprintf('Ok, starting...\n'),close all

pyff('setdir','');
% pyff('setdir','basename',basename);
pyff('play'); pause(5)
stimutil_waitForMarker('stopmarkers',RUN_END);
pyff('stop');
pyff('quit');
fprintf(['Practice finished!\n'])

%% Hauptexperiment
nBlocks = 10;

for ii=1:nBlocks
  
  if ii==1, log_filename = [TODAY_DIR basename VP_CODE '.log'];
  else log_filename = [TODAY_DIR basename VP_CODE num2str(ii) '.log']; end

  fprintf(['Press <RETURN> to start block ' num2str(ii)  '.\n']); pause;
  fprintf('Ok, starting...\n'), close all
  setup_alpha_feedback;
  pyff('setint','nr_trials',nTrials/nBlocks);
  pyff('setdir','basename',basename);

  pyff('play'); pause(5)
  stimutil_waitForMarker({['S' num2str(RUN_END)]});


  fprintf(['Block ' num2str(ii) ' finished!\n'])
  pyff('stop');
  pyff('quit');
end


%%
if ~strcmp(VP_CODE, 'Temp');
  save(vp_counter_file, 'vp_number');
end
fprintf(['Experiment finished!\n'])