clear all; close all; clc;

DATADIR = 'WHERE\IS\DATA';
SAVEDIR = 'WHERE\TO\SAVE';
%% SSVEP
SSVEPDATA = 'EEG_SSVEP.mat';
STRUCTINFO = {'EEG_SSVEP_train', 'EEG_SSVEP_test'};
SESSIONS = {'session1', 'session2'};
TOTAL_SUBJECTS = 54;
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
dataLength = (TOTAL_SUBJECTS * 2 * length(STRUCTINFO));
conSMT = cell(1, dataLength);
idx = 1;
%%
fprintf('Visualization tutorial\n')
for sessNum = 1:length(SESSIONS)
    session = SESSIONS{sessNum};
    for subNum = 1:TOTAL_SUBJECTS
        subject = sprintf('s%d',subNum);
        data = importdata(fullfile(DATADIR,session,subject,SSVEPDATA));
        for info = 1:length(STRUCTINFO)
            CNT = data.(STRUCTINFO{info});
            CNT = rmfield(CNT, 'smt');
            CNT = prep_resample(CNT, FS, {'Nr', 0});
            CNT = prep_filter(CNT, {'frequency', FREQBAND});
            SMT = prep_segmentation(CNT, {'interval', SEGTIME});
            SMT = prep_average(SMT);            
            
            conSMT{idx} = SMT;
            
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

vis_plotController(avSMT, [], params);
mkdir(SAVEDIR);
saveas(gcf, fullfile(SAVEDIR, 'SSVEP_GA'), 'fig');