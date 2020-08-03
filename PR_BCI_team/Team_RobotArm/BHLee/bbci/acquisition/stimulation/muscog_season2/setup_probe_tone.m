%contains options for probe tone stimulation (for probe_tone_exp.m and
%stim_probe_tone.m
% 
% I.Sturm 15/2009

global STIM_DIR VP_SCREEN
%STIM_DIR='C:\users\irene\BCI\matlab\svn\bbci\investigation\personal\irene\music_cognition\Sounds\Stimuli\'
STIM_DIR= [BCI_DIR 'acquisition\data\sound\muscog\'];

opt = struct;
opt.stim_type = 'shepard';
opt.min_rep = 7;
opt.max_rep = 11;
opt.duration = 0.35;
opt.aso = 0.41;
opt.fs = 44100;
opt.fade = 0.05;
opt.major_or_minor = 'major';


opt.bv_host= 'localhost';
%opt.position= [5 200 640 480];  %% for testing
opt.position= VP_SCREEN;
opt.cross= 1;
opt.countdown = 5;
opt.handle_background= stimutil_initFigure(opt);
opt.break_duration=15;



% generate order of stimuli and load sounds
%[order sounds_key]= load_mimu(opt);


%