clear all; close all; clc;

DATADIR = 'WHERE\IS\DATA';
SAVEDIR = 'WHERE\TO\SAVE';
%% MI
MOTORDATA = 'EEG_MI.mat';
STRUCTINFO = 'EEG_MI_train';
SESSION = 'session1';
SUBJECT = 's1';
%% INITIALIZATION
FS = 100;
FREQBAND = [8 12];
SEGTIME = [-1000 4000];
BASELINE = [-500 0];

params = {'Interval', [-1000 0; 1000 1300; 1500 1800; 2000 2300;];
    'Channels', {'C3', 'C4'};
    'Class', {'right'; 'left'};
    'TimePlot', 'on';
    'TopoPlot', 'on'; 
    'rValue', 'on'; 
    'Range', 'mean';
    'Patch', 'on';
    'Quality', 'high'; 
    'baseline', BASELINE};
%%
fprintf('Visualization tutorial\n')

data = importdata(fullfile(DATADIR,SESSION,SUBJECT,MOTORDATA));
CNT = data.(STRUCTINFO);
CNT = rmfield(CNT, 'smt');
CNT = prep_resample(CNT, FS, {'Nr', 0});
CNT = prep_filter(CNT, {'frequency', FREQBAND});
SMT = prep_segmentation(CNT, {'interval', SEGTIME});
SMT = prep_envelope(SMT);
SMT = prep_baseline(SMT, {'Time', BASELINE});

avSMT = prep_average(SMT);
rSMT = proc_signedrSquare(SMT);

vis_plotController(avSMT,rSMT, params);

mkdir(SAVEDIR);
saveas(gcf, fullfile(SAVEDIR, sprintf('%s_MI',SUBJECT)), 'fig');
