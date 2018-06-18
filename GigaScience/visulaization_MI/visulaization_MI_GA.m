clear all; close all; clc;

DATADIR = 'WHERE\IS\DATA';
SAVEDIR = 'WHERE\TO\SAVE';
%% MI
MOTORDATA = 'EEG_MI.mat';
STRUCTINFO = {'EEG_MI_train', 'EEG_MI_test'};
SESSIONS = {'session1', 'session2'};
TOTAL_SUBJECTS = 54;
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
dataLength = (TOTAL_SUBJECTS * 2 * length(STRUCTINFO));
conSMT = cell(1, dataLength);
conrSMT = cell(1, dataLength);
idx = 1;
%%
fprintf('Visualization tutorial\n')
for sessNum = 1:length(SESSIONS)
    session = SESSIONS{sessNum};
    for subNum = 1:TOTAL_SUBJECTS
        subject = sprintf('s%d',subNum);
        data = importdata(fullfile(DATADIR,session,subject,MOTORDATA));
        for info = 1:length(STRUCTINFO)            
            CNT = data.(STRUCTINFO{info});
            CNT = rmfield(CNT, 'smt');
            CNT = prep_resample(CNT, FS, {'Nr', 0});
            CNT = prep_filter(CNT, {'frequency', FREQBAND});
            SMT = prep_segmentation(CNT, {'interval', SEGTIME});
            SMT = prep_envelope(SMT);
            SMT = prep_baseline(SMT, {'Time', BASELINE});
            
            conSMT{idx} = prep_average(SMT);
            conrSMT{idx} = proc_signedrSquare(SMT);
            
            fprintf('%06.2f%%...', idx/dataLength*100);
            if mod(idx, 20) == 0
                fprintf('\n');
            end            
            idx = idx + 1;
        end
    end
end
fprintf('\n');

avSMT = grandAverage_prototye(conSMT);
avrSMT = grandAverage_prototye(conrSMT);

vis_plotController(avSMT,avrSMT, params);

mkdir(SAVEDIR);
saveas(gcf, fullfile(SAVEDIR, 'MI_GA'), 'fig');
close gcf;
