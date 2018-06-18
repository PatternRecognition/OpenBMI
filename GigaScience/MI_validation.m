%% MI validation
% Initialization
session = {'session1', 'session2'};
dir = 'G:\DB\';
fs=100;
totalNUM=54;

general_initparam = { 'task',{'mi_off','mi_on'}; ...
    'channel_index', [8 9 10 11 13 14 15 18 19 20 21 33 34 35 36 37 38 39 40 41]; ...
    'band', [8 30]; ...
    'time_interval', [1000 3500]; ...
    'CSPFilter', 2; ...
    };

% for MI_cv
Niteration = 10; 
% for CSSP
tau=[0.01:0.01:0.15]*1000; 
% for FBCSP
filterbank = [4 8;8 12;12 16;16 20;20 24;24 28;28 32;32 36;36 40]; 
NUMfeat = 4; 
% for BSSFO
bssfo_param = {'init_band', [4 40]; ...
    'numBands', 30; ...
    'numIteration', 10; ...
    'mu_band', [7 15]; ...
    'beta_band', [14 30]; ...
    };
%% validation
for sess = 1:length(session)
    for sub = 1:totalNUM
        fprintf('%d-th ...\n',sub);
        snum = num2str(sub);
        filetrain = fullfile([dir, session{sess},'\s',snum,'\EEG_MI.mat']);
        load(filetrain);
        CNT{1} = prep_resample(EEG_MI_train, fs,{'Nr', 0});
        CNT{2} = prep_resample(EEG_MI_test, fs,{'Nr', 0});
        ACC.MI_cv(sub,sess) = mi_performance(CNT,general_initparam,Niteration);
        ACC.MI_off2on(sub,sess) = mi_performance_off2on(CNT,general_initparam);     
        ACC.MI_CSSP(sub,sess) = cssp_off2on(CNT,general_initparam,tau);
        ACC.MI_FBCSP(sub,sess) = fbcsp_off2on(CNT,general_initparam,filterbank,NUMfeat);
        ACC.MI_BSSFO(sub,sess) = bssfo_off2on(CNT,general_initparam,bssfo_param);
        fprintf('CSP_crsval = %f\n',ACC.MI_cv(sub,sess));
        fprintf('CSP = %f\n',ACC.MI_off2on(sub,sess));
        fprintf('CSSP = %f\n',ACC.MI_CSSP(sub,sess));
        fprintf('FBCSP = %f\n',ACC.MI_FBCSP(sub,sess));
        fprintf('BSSFO = %f\n',ACC.MI_BSSFO(sub,sess));
        clear CNT EEG_MI_test EEG_MI_train filetrain Dat1 Dat2
    end
end
