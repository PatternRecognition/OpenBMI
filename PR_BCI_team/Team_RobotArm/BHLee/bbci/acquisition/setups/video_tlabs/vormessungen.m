
%% Impedanzcheck
bvr_sendcommand('checkimpedances');
fprintf('Prepare cap. Press <RETURN> when finished.\n');
pause
fprintf('Unplug light diode. Press <RETURN> when finished.\n');
pause

%% Experimental settings
VEP_file = [BCI_DIR 'acquisition/setups/video_tlabs/VEP_fb'];
Oddball_file = [BCI_DIR 'acquisition/setups/video_tlabs/Oddball_fb'];


%% Artifacts
vps = VP_SCREEN;
VP_SCREEN = VP_SCREEN-[0 175 0 0];
[seq, wav, opt]= setup_season10_artifacts_demo('clstag', '');
fprintf('Press <RETURN> to start TEST artifact measurement.\n');
pause; fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt, 'test',1);
[seq, wav, opt]= setup_season10_artifacts('clstag', '');
fprintf('Press <RETURN> to start artifact measurement.\n');
pause; fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt);
close all

%% Relaxation
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_season10_relax;
fprintf('Press <RETURN> to start RELAX measurement.\n');
pause; fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt);
close all
VP_SCREEN = vps;


%% Startup pyff
pyff('startup','gui',0,'dir','C:/svn/pyff/src');
bvr_sendcommand('viewsignals');
pause(5)
send_xmlcmd_udp('init', '127.0.0.1', 12345);

%% Vormessung: VEP
pyff('init','CheckerboardVEP');pause(5)
pyff('load_settings', VEP_file);
pyff('setint','screen_pos',VP_SCREEN);
fprintf('Press <RETURN> to start VEP measurement.\n'); pause;
fprintf('Ok, starting...\n'),close all
% pyff('setdir', '');
pyff('setdir', 'basename', 'vep');
pyff('play')
stimutil_waitForMarker('S255');
fprintf('VEP measurement finished.\n')
bvr_sendcommand('stoprecording');
pyff('stop');
pyff('quit');
fprintf('Press <RET> to continue.\n'); pause;

%% Vormessung: Oddball (visual)
pyff('init','VisualOddball');pause(5)
pyff('load_settings', Oddball_file);
fprintf('Press <RETURN> to start Oddball experiment.\n'); pause;
fprintf('Ok, starting...\n'),close all
pyff('setdir', 'basename', 'oddball');
pyff('play')
% wait until end of feedback
% state= acquire_bv(1000, 'localhost');
% state.reconnect= 1;
% stimutil_waitForMarker('S255','bv_bbciclose',0,'state',state);
stimutil_waitForMarker('S255');
fprintf('Oddball finished.\n')
bvr_sendcommand('stoprecording');
pyff('stop');
pyff('quit');


fprintf('Type ''run_video'' and press <RET>\n');
