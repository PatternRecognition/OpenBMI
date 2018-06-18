clear all; close all; clc;

DATADIR = 'WHERE\IS\DATA';
SAVEDIR = 'WHERE\TO\SAVE';
%% SSVEP
SSVEPDATA = 'EEG_SSVEP.mat';
STRUCTINFO = 'EEG_SSVEP_train';
SESSION = 'session1';
SUBJECT = 's1';
%% INITIALIZATION
FS = 100;
FREQBAND = [.5 40];
SEGTIME = [0 4000];

params = {
    'Channels', {'Oz'};
    'Class', {'up'; 'left'; 'right'; 'down'};
    'FFTPlot', 'on';
    };
%%
fprintf('Visualization tutorial\n')
data = importdata(fullfile(DATADIR,SESSION,SUBJECT,SSVEPDATA));
CNT = data.(STRUCTINFO);
CNT = rmfield(CNT, 'smt');
CNT = prep_resample(CNT, FS, {'Nr', 0});
CNT = prep_filter(CNT, {'frequency', FREQBAND});
SMT = prep_segmentation(CNT, {'interval', SEGTIME});
avSMT = prep_average(SMT);

vis_plotController(avSMT, [], params);
mkdir(SAVEDIR);
saveas(gcf, fullfile(SAVEDIR, sprintf('%s_SSVEP',SUBJECT)), 'fig');