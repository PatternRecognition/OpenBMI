% Load data, but without EOG channels
file='VPsah_09_03_16/visual_p300_hex_targetVPsah';
[cnt,mrk,mnt]= eegfile_loadMatlab(file, 'clab',{'not','E*'});

% High-pass filter at 0.5Hz (optional)
%b= procutil_firlsFilter(0.5, cnt.fs);
%cnt= proc_filtfilt(cnt, b);
  
% Segmentation
epo= cntToEpo(cnt, mrk, [-200 800]);
  
% Artifact rejection based on maxmin difference criterion on frontal chans
crit_maxmin= 100;
epo= proc_rejectArtifactsMaxMin(epo, crit_maxmin, ...
            'clab','F3,z,4', 'ival',[-200 800], 'verbose',1);

% Baseline subtraction, and calculation of a measure of discriminability
epo= proc_baseline(epo, [-200 0]);
epo_r= proc_r_square_signed(epo);

% Select some discriminative intervals
fig_set(1);
[cfy_ival, nfo]= ...
    select_time_intervals(epo_r, 'visualize', 1, 'visu_scalps', 1);

fig_set(3)
H= grid_plot(epo, mnt, defopt_erps);
grid_addBars(epo_r, 'h_scale',H.scale);

fig_set(2);
H= scalpEvolutionPlusChannel(epo, mnt, {'Cz','PO8'}, cfy_ival, defopt_scalp_erp2);
grid_addBars(epo_r);

fig_set(4, 'shrink',[1 2/3]);
scalpEvolutionPlusChannel(epo_r, mnt, {'Cz','PO8'}, cfy_ival, defopt_scalp_r2);


% Classification

fv= proc_jumpingMeans(epo, cfy_ival);
xvalidation(fv, 'LDA', 'loss','classwiseNormalized');

xvalidation(fv, 'NCC', 'loss','classwiseNormalized');


C_LDA= trainClassifier(fv, 'LDA');
ff= proc_flaten(fv)
fv_LDA= ff;
fv_LDA.x= C_LDA.w'*ff.x;

plotOverlappingHist(fv_LDA, 100);

fig_set(2)
C_NCC= trainClassifier(fv, 'NCC');
ff= proc_flaten(fv)
fv_NCC= ff;
fv_NCC.x= C_NCC.w'*ff.x;

plotOverlappingHist(fv_NCC, 100);
