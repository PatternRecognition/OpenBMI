%% Prepare offline calibration
word_list= {'WINKT_','QUARZ_','FJORD_'};
% word_list= {'FJORD_'};
all_letters = [word_list{:}];
practice_letters = 'TOPO';

% word_list = 'BLA';

%% Calibration **practice**
setup_speller
fprintf('Press <RETURN> to start %s PRACTICE.\n',cs); pause;
pyff('setdir','');

pyff('set','offline',1,'desired_phrase',practice_letters);
fprintf('Ok, starting...\n'),close all

pyff('play');
pause(5)
stimutil_waitForMarker({['S' num2str(COPYSPELLING_FINISHED)] ['S' num2str(RUN_END)] 'R  8'});

pyff('stop'); pause(1)
pyff('quit');


%% Calibration **RUN**
fprintf('Press <RETURN> to start %s calibration.\n',cs); pause;
setup_speller 
pyff('set','offline',1,'desired_phrase',all_letters);
pyff('setdir','basename',[calib_prefix cs '_']);
fprintf('Ok, starting...\n'),close all
pyff('play');
pause(5)
stimutil_waitForMarker({['S' num2str(COPYSPELLING_FINISHED)] ['S' num2str(RUN_END)]  'R  8'});

fprintf('Calibration finished.\n')
pyff('stop');
pyff('quit');
