bvr_sendcommand('checkimpedances');
fprintf('Prepare cap. Press <RETURN> when finished.\n');
pause

%% Artifacts demo
setup_fasor_LCT2_artifacts_demo;
fprintf('Press <RETURN> when ready to start artifact measurement test.\n');
pause
stim_artifactMeasurement(seq, wav, opt, 'test',1);

%% Artifacts Measurement
setup_fasor_LCT2_artifacts;
fprintf('Press <RETURN> when ready to start artifact measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt);

%% KSS
fprintf('Ask the subject the KSS; Press <RETURN> when ready to go to the next run.\n');pause
pause

%% Durchgänge  %ev. früher abbrechen, falls VP erschöpft
for k = 1:6
  
  setup_fasor_lct2_laser_fixCross;

%% K0 demo, nur 1. Durchgang
 if k == 1        
   fprintf('Press <RETURN> when ready to start the test of K0.\n');
   pause
   stim_laserCues_BackgroundSound(stim, opt, 'test',1)
 end
  
%% K0
 fprintf('Press <RETURN> when ready to start the next run: K0.\n');
 pause
 stim_laserCues_BackgroundSound(stim, opt);

%% KSS
 fprintf('Ask the subject the KSS; Press <RETURN> when ready to go to the next run.\n');pause
 pause

 setup_fasor_lct2_laser_selbstFahren;

%% K2 demo, nur 1. Durchgang
 if k == 1        
   fprintf('Press <RETURN> when ready to start the demo of K2.\n');   
   fprintf('VL: <RETURN> druecken und danach Simulator starten in D:\Fasor\LCT_langweilig_v2\mbc_pc_logitech.exe')
   pause
   stim_laserCues_LCT_Simulation(stim, opt, 'test',1);
   fprintf('VL: Simulator mit <ESC> stoppen')
 end

%% K2
 fprintf('Press <RETURN> when ready to start the next run: K2.\n');
 fprintf('VL: <RETURN> druecken und danach Simulator starten in D:\Fasor\LCT_langweilig_v2\mbc_pc_logitech.exe')
 pause 
 stim_laserCues_LCT_Simulation(stim, opt);
 fprintf('VL: Simulator mit <ESC> stoppen')

%% KSS
 fprintf(['Ask the subject the KSS; Press <RETURN> when ready to go to the next run, Durchgang ' int2str(k) '.\n']);
 pause
 
end
 
ppTrigger(0);
