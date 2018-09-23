%% signals plot in time
data = importdata('D:\DB_tree\session1\s1\EEG_P300.mat');
CNT = data.EEG_P300_train;

segtime = [3000 3006]; %ms¥‹¿ß

XLIM = segtime * CNT.fs / 1000;

bias = max(abs(reshape(CNT.x, 1, [])));
bias = cumsum(repmat(bias, 1, length(CNT.chan)));

chVec = flip(CNT.x, 2);
chVec = chVec + bias;

plot(1:length(chVec), chVec,'k');
xlim([1 length(chVec)]);
ylim([0 sum(minmax(bias))]);
yticks(bias);
yticklabels(flip(CNT.chan));

yticks(flip(bias))
yticklabels(flip(CNT.chan))
