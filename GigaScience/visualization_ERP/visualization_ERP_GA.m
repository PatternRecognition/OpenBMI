clear all; close all; clc;

DATADIR = 'WHERE\IS\DATA';
SAVEDIR = 'WHERE\TO\SAVE';
%% ERP
ERPDATA = 'EEG_ERP.mat';
STRUCTINFO = {'EEG_ERP_train', 'EEG_ERP_test'};
SESSIONS = {'session1', 'session2'};
TOTAL_SUBJECTS = 54;
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
        data = importdata(fullfile(DATADIR,session,subject,ERPDATA));
        for info = 1:length(STRUCTINFO)
            CNT = data.(STRUCTINFO{info});
            CNT = rmfield(CNT, 'smt');
            CNT = prep_resample(CNT, FS, {'Nr', 0});
            CNT = prep_filter(CNT, {'frequency', FREQBAND});
            SMT = prep_segmentation(CNT, {'interval', SEGTIME});
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

avSMT = grandAverage_prototye(conSMT, true);
rSMT = grandAverage_prototye(conrSMT, true);

vis_plotController(avSMT, rSMT, params);

mkdir(SAVEDIR)
saveas(gcf, fullfile(SAVEDIR, 'ERP_GA'), 'fig');