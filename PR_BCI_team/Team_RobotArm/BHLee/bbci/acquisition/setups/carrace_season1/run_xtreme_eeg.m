%% Brake oddball test
setup_xtreme_eeg_braking;
fprintf('Press <RETURN> to TEST brake-oddball.\n');
pause; fprintf('Ok, starting...\n');
stim_oddballVisual(10, opt, 'test',1);
fprintf('Press <RETURN> to start brake-oddball.\n');
pause; fprintf('Ok, starting...\n');
stim_oddballVisual(N, opt);

%% Brake oddball with EEG
setup_carrace_season1_braking;
fprintf('Press <RETURN> to start brake-oddball.\n');
pause; fprintf('Ok, starting...\n');
stim_oddballVisual(N, opt);
pause(2);

%% Self paced breaking with EEG
stimutil_fixationCross;
fprintf('Starting Local InputReader.\n');
fprintf('Ok.\n');
fprintf('Press <RETURN> to start self-paced braking.\n');
pause; fprintf('Ok, starting...\n');
dos('C:\bbci\torcs-1.3.1\runtime\InputReader.exe local BreakSync C:\bbci\torcs-1.3.1\runtime\ &');
pause(1);
cd(TODAY_DIR)
bvr_startrecording('selfpaced_braking');
pause(6*60 + 10);
bvr_sendcommand('stoprecording');
fprintf('Stop InputReader by pressing <CTRL-C> in the corresponding window.\n');
fprintf('Press <RETURN> to continue.\n');
pause;

%-Starting torcs
fprintf('Starting Torcs.\n');
cd(TORCS_DIR)
dos('wtorcs.exe &');
cd(TODAY_DIR)

%-newblock
fprintf('Start demo mode of TORCS (see readme file).\n');
fprintf('Press <RETURN> to start carrace observation.\n');
pause; fprintf('Ok, starting...\n');
bvr_startrecording('carrace_observation');
pause(4*60);
bvr_sendcommand('stoprecording');
pause(5);

%-Startign server InputReader
fprintf('Starting Server InputReader.\n');
dos('C:\bbci\torcs-1.3.1\runtime\InputReader.exe server &');
fprintf('Ok.\n');

%-newblock
fprintf('Start player mode of TORCS (see readme file).\n');
fprintf('Press <RETURN> to start carrace run 1.\n');
pause; fprintf('Ok, starting...\n');
bvr_startrecording('carrace_drive');
pause(45*60);
bvr_sendcommand('stoprecording');
pause(5);


%-newblock
fprintf('Start player mode of TORCS (see readme file).\n');
fprintf('Press <RETURN> to start carrace run 2.\n');
pause; fprintf('Ok, starting...\n');
bvr_startrecording('carrace_drive');
pause(45*60);
bvr_sendcommand('stoprecording');
pause(5);

%-newblock
fprintf('Start player mode of TORCS (see readme file).\n');
fprintf('Press <RETURN> to start carrace run 3.\n');
pause; fprintf('Ok, starting...\n');
bvr_startrecording('carrace_drive');
pause(45*60);
bvr_sendcommand('stoprecording');
pause(5);
