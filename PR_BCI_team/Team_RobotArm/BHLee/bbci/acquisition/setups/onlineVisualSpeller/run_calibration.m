%% Prepare offline calibration
% Triggers
COPYSPELLING_FINISHED = 246;
RUN_END = 253;

word_list= {'WINKT','QUARZ_','FJORD_', 'SPHINX_' , 'MEMME_'};
word_list= {'WINKT','QUARZ_','FJORD'};
all_letters = [word_list{:}];
practice_letters = 'TOPO';

% word_list = 'BLA';

%% Calibration **practice**
setup_speller
fprintf('Press <RETURN> to start %s PRACTICE.\n',currentSpeller); pause;
pyff('setdir','');

pyff('set','offline',1,'desired_phrase',practice_letters);
fprintf('Ok, starting...\n'),close all

pyff('play');
pause(5)
stimutil_waitForMarker({['S' num2str(COPYSPELLING_FINISHED)] ['S' num2str(RUN_END)]});

pyff('stop'); pause(1)
pyff('quit');


%% Calibration **RUN**
fprintf('Press <RETURN> to start %s calibration.\n',currentSpeller); pause;
setup_speller 
pyff('set','offline',1,'desired_phrase',all_letters);
pyff('setdir','basename',[calib_prefix currentSpeller '_']);
fprintf('Ok, starting...\n'),close all
pyff('play');
pause(5)
stimutil_waitForMarker({['S' num2str(COPYSPELLING_FINISHED)] ['S' num2str(RUN_END)]});

fprintf('Calibration finished.\n')
pyff('stop');
bvr_sendcommand('stoprecording');
pyff('quit');
