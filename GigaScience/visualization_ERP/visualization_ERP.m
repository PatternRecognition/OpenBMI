clear all; close all; clc;

DATADIR = 'WHERE\IS\DATA';
SAVEDIR = 'WHERE\TO\SAVE';
%% SSVEP
ERPDATA = 'EEG_ERP.mat';
STRUCTINFO = 'EEG_ERP_train';
SESSION = 'session1';
SUBJECT = 's1';
%% INITIALIZATION
FS = 100;
FREQBAND = [.5 40];
SEGTIME = [-200 800];
BASELINE = [-200 0];

params = {
    'Interval', [-100, 0; 150 200; 240, 270;320 350]; 
    'Channels', {'Oz', 'Cz'}; 
    'Class', {'target'; 'nontarget'};
    'TimePlot', 'on'; 
    'TopoPlot', 'on'; 
    'rValue', 'on';
    'Range', [-1 2];  
    'Patch', 'on'; 
    'Quality', 'high'; 
    'baseline', BASELINE;
    };
%%
fprintf('Visualization tutorial\n')
data = importdata(fullfile(DATADIR,SESSION,SUBJECT,ERPDATA));
CNT = data.(STRUCTINFO);
CNT = rmfield(CNT, 'smt');
CNT = prep_resample(CNT, FS, {'Nr', 0});
CNT = prep_filter(CNT, {'frequency', FREQBAND});
SMT = prep_segmentation(CNT, {'interval', SEGTIME});
SMT = prep_baseline(SMT, {'Time', BASELINE});

avSMT = prep_average(SMT);
rSMT = proc_signedrSquare(SMT);

vis_plotController(avSMT, rSMT, params);

saveas(gcf, fullfile(SAVEDIR, sprintf('%s_ERP',SUBJECT)), 'fig');