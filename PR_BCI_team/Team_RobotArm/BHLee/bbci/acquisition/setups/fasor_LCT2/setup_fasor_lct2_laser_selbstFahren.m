stim= [];
stim.cue= struct('classes', {'left'});
[stim.cue.marker]= deal(64);
[stim.cue.nEvents]= deal(120); % 120 trials sind ungefähr 20 Minuten
[stim.cue.timing]= deal([10000 500 0]);  % [InterTrialInterval LaserDuration CrossOff]
[stim.cue.jitter]= deal(5000);
stim.msg_intro= 'REAKTIONSAUFGABE und FAHREN (ca. 20 Min)';

opt= [];
opt.handle_background= stimutil_initFigure(opt);
[H, opt.handle_cross]= stimutil_cueArrows({stim.cue.classes}, opt, 'cross',1);
Hc= num2cell(H);
stim.cue.handle=deal(Hc{:});

desc= stimutil_readDescription('fasor_LCT2_K2');
h_desc= stimutil_showDescription(desc, 'waitfor',0);

opt.filename= 'fasor_LCT2_K2_';
opt.msg_fin= 'Ende dieses Blocks';
opt.delete_obj= h_desc.axis;
opt.bv_host= ''; % 

fprintf('VERSUCHSLEITER bitte jetzt: \n');
%fprintf('  Bildschirm der VP zum primären Windows-Bildschirm machen (für LCT Simulator!)\n');
fprintf('  Start von LCT Simulator vorbereiten (D:\\Fasor\\LCT_langweilig_v2\\mbc_pc_logitech.exe)\n');
fprintf('  LaserStim testen mit :\n  stim_laserCues_LCT_Simulation(stim, opt, ''test'',1);\n');
fprintf('  Sonst :\n  stim_laserCues_LCT_Simulation(stim, opt);\n');
fprintf('  LCT Simulator starten (D:\\Fasor\\LCT_langweilig_v2\\mbc_pc_logitech.exe) starten.\n');
fprintf('  Nach Ende der Trials LCT Simulator beenden (ESC)\n');
