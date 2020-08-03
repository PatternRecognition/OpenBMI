acqFolder = [BCI_DIR 'acquisition/setups/' session_name '/'];
CENTERSPELLER_file = [acqFolder 'CenterSpeller_feedback'];
ODDBALL_file = [acqFolder 'Oddball_feedback'];

pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'], 'bvplugin', 0);
fprintf('Starting Pyff...\n'); pause(10);


% Visual Speller
condition_tags= {'Cond1','Cond2','Cond3'};
order= perms(1:length(condition_tags));
conditionsOrder= uint8(order(1+mod(VP_NUMBER-1, size(order,1)),:));

%% main loop
for jj= conditionsOrder,
  switch(condition_tags{jj}),
    case 'Cond1',
      % ...
    case 'Cond2',
      % ...
    case 'Cond3',
      % ...
  end
  % common code for all conditions
  bvr_startrecording(['tactile_' condition_tags{jj}], 'impedances',0);
  % start stimulation
  % ...
  bvr_sendcommand('stoprecording');
end  


%% --- - --- Session finished

acq_vpcounter(session_name, 'close');
fprintf('Session %s finished.\n', session_name);
  




  

%
%% ----- Oddball - 4 different ISIs, randomized order
%


%% Oddball - Practice
pyff('init', 'VisualOddballVE'); pause(.5);
pyff('load_settings', ODDBALL_file);
pyff('setint','nTrials',10);
stimutil_waitForInput('msg_next','to start Oddball practice.');
pyff('play');
stimutil_waitForMarker(RUN_END);
fprintf('Oddball practice finished.\n')
pyff('quit');

%% Oddball - Recording
pyff('init', 'VisualOddballVE'); pause(.5);
pyff('load_settings', ODDBALL_file);
stimutil_waitForInput('msg_next','to start Oddball recording.');
pyff('play', 'basename', 'Oddball', 'impedances', 0);
stimutil_waitForMarker(RUN_END);
fprintf('Oddball recording finished.\n')
pyff('quit');
fprintf('Press <RETURN> to continue.\n'); pause;

