%% Artifacts
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


%% Startup pyff
oldp = pwd;
cd D:\svn\pyff\src
if TEST
  warning('Pyff test option on!')
  keyboard
  pyff('startup','gui',0,'bvplugin',0);
else
  pyff('startup','gui',0);
end
cd(oldp);
bvr_sendcommand('viewsignals');
pause(5)
send_xmlcmd_udp('init', '127.0.0.1', 12345);
VP_SCREEN = [-BigScreen(1) 0 BigScreen];

%% Vormessung: VEP  (darken room!)
pyff('init','CheckerboardVEP');pause(5)
pyff('load_settings', VEP_file);
pyff('setint','screen_pos',VP_SCREEN);
fprintf('Press <RETURN> to start VEP measurement.\n'); pause;
fprintf('Ok, starting...\n'),close all
pyff('setdir', 'basename', 'vep');
pause(5)
pyff('play')
pause(5)
stimutil_waitForMarker('S255');
fprintf('VEP measurement finished.\n')
bvr_sendcommand('stoprecording');
pyff('stop');
pyff('quit');
fprintf('Press <RET> to continue.\n'); pause;


%% Vormessung: Oddball (visual)
% Seltener Stimulus: rotes Quadrat; Häufiger Stimulus: grüner Kreis
% Aufgabe: Zähle die Anzahl der roten Quadrate während des ganzen Experiments

pyff('init','VisualOddball');pause(5)
pyff('load_settings', Oddball_file);
pyff('setint','screen_pos',VP_SCREEN);
fprintf('Press <RETURN> to start Oddball experiment.\n'); pause;
fprintf('Ok, starting...\n'),close all
pyff('setdir', 'basename', 'oddball');
pause(5)
pyff('play')
pause(5)
stimutil_waitForMarker('S255');
fprintf('Oddball finished.\n')
bvr_sendcommand('stoprecording');
pyff('stop');
pyff('quit');


