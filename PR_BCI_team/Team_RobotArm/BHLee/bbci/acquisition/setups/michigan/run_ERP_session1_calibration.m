error('obsolete')

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
setup_ERP_speller
fprintf('Press <RETURN> to start %s PRACTICE.\n','HexoSpellerVE'); 
pause;

fbint.offline= 1;
fb.desired_phrase= practice_letters;
pyff('set',fb)
pyff('setint',fbint)
fprintf('Ok, starting...\n'),close all

pyff('play');
pause(5)
stimutil_waitForMarker({['S' num2str(COPYSPELLING_FINISHED)] ['S' num2str(RUN_END)]});

pyff('stop'); 
pause(1)
pyff('quit');


%% Calibration **RUN**
fprintf('Press <RETURN> to start %s calibration.\n','HexoSpellerVE'); pause;
setup_ERP_speller 
fb.desired_phrase= all_letters;
pyff('set',fb)
pyff('setint',fbint);
fprintf('Ok, starting...\n');
close all
pyff('play','basename',[calib_prefix 'HexoSpellerVE_']);
pause(5)
stimutil_waitForMarker({['S' num2str(COPYSPELLING_FINISHED)] ['S' num2str(RUN_END)]});

fprintf('Calibration finished.\n')
pyff('stop');
pyff('quit');
