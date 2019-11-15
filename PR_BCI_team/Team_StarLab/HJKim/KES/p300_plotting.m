function [out] = p300_plotting(file, marker)
segTime=[-200 1000];
baseTime=[-200 0];
selTime=[0 1000];
nFeature=10;
Freq=[0.4 40];

% marker={'1','target';'2','nontarget'};
% marker={'1','target';'77','nontarget'};
% marker={'1',1;'2',2;'3',3;'4',4;'5',5;'6',6;'7',7;'8',8;'9',9;'10',10;'11',11;'12',12};
fs=100; 
field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};

[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker',marker});
cnt=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
% ch_idx = [1, 3];
% cnt=prep_selectChannels(cnt,{'Name',{'Cz', 'Oz'}});
cnt=prep_resample(cnt, fs);
% cnt=prep_selectChannels(cnt,{'Index',[3 ]});
% cnt.x=notch_filter(cnt.x, 100, [8 12]);
cnt=prep_filter(cnt, {'frequency', Freq});

% %%%
% mask = false(1,1980);
% mask(1:60:1980) = true;
% cnt_test = cnt;
% cnt_test.t(mask) = cnt_test.t(mask) + ceil(116/(10000/fs))-9;
% cnt_test.t(~mask) = cnt_test.t(~mask) + ceil(1311/(10000/fs))-9;%%%

smt=prep_segmentation(cnt, {'interval', segTime});
% smt=prep_envelope(smt);
% smt = prep_selectTrials(smt, {'Index', [
smt=prep_baseline(smt, {'Time',baseTime});
% smt=prep_selectTrials(smt, {'Index', [1:500]});
plot_x = -200:1000/fs:1000;
out = smt;
smt = prep_average(smt);
figure;
YLIM = minmax(reshape(smt.x, 1, []));
k = 1;
for i = [7 8 9 12 13 14 15 16 19 20 21 31]
    subplot(3,4,k); plot(smt.ival, smt.x(:,:,i)); 
    legend(smt.class(:,2)); 
    title(smt.chan(i)); ylim(YLIM);
    k = k + 1;
end

% subplot(16,1,1); plot(smt.ival, smt.x(:,:,1)); legend(smt.class(:,2)); title(smt.chan(1)); ylim(YLIM);
% subplot(2,1,2); plot(smt.ival, smt.x(:,:,2)); legend(smt.class(:,2)); title(smt.chan(2)); ylim(YLIM);

% target = smt.x(:,smt.y_logic(1,:),:);
% n_target = smt.x(:,smt.y_logic(2,:),:);
% 
% mean_target = squeeze(mean(mean(target, 3),2));
% mean_n_target = squeeze(mean(mean(n_target, 3),2));
% 
% plot(0:10:800, mean_target); hold on; plot(0:10:800, mean_n_target);
