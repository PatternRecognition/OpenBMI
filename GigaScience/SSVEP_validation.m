%% SSVEP validation
% initialization
session = {'session1', 'session2'};
dir = 'G:\DB\';
fs=100;
totalNUM=54;

%init
initParam = {'time', 4;...
'freq' , [5, 7, 9, 11];...
'fs' , 100;...
'band' ,[0.5 40];...
'channel_index', [23:32]; ...
'time_interval' ,[0 4000]; ...
'marker',  {'1','up';'2', 'left';'3', 'right';'4', 'down'}; ...
};

%% validation
for sess = 1:length(session)
    for sub = 1:totalNUM
        fprintf('%d-th ...\n',sub);
        snum = num2str(sub);
        filetrain = fullfile([dir, session{sess},'\s',snum,'\EEG_SSVEP.mat']);
        load(filetrain);
        CNT{1} = prep_resample(EEG_SSVEP_train, fs,{'Nr', 0});
        CNT{2} = prep_resample(EEG_SSVEP_test, fs,{'Nr', 0});
        ACC.SSVEP(sub,sess) = ssvep_performance(CNT, initParam);
        fprintf('%d = %f\n',sub, ACC.SSVEP(sub,1));
        clear CNT EEG_SSVEP_train EEG_SSVEP_test filetrain Dat1 Dat2
    end
end



