file= 'Gabriel_00_09_05/selfpaced2sGabriel';

%% options for plotting
opt= struct('colorOrder', [1 0 0; 0 0.7 0; 0 0 1; 0.9 0.8 0]);
opt.xGrid= 'on';
opt.yZeroLine= 'on';
opt.resolution= 32;
opt.shading= 'flat';



[cnt, mrk, mnt]= loadProcessedEEG(file);

%% find intervals of active experiment, i.e., without break
%% periods in which the subject might produce strong artifacts.
blk_ival= getActivationAreas(cnt.title);
blk= struct('fs',cnt.fs, 'ival',blk_ival');
[cnt_active, blk, mrk_active]= proc_concatBlocks(cnt, blk, mrk);
cnt_active= proc_selectChannels(cnt_active, 'not','E*');

%% simple high-pass filter
cnt_active= proc_subtractMovingAverage(cnt_active, 3000, 'centered');

%% apply TDSEP
tau= 0:5;
W= tdsep0(cnt_active.x', tau);

epo= makeEpochs(cnt_active, mrk_active, [-1200 600]);
plotPatternsPlusSpecERP(epo, mnt, W, opt);
%% plot without spectra
plotPatternsPlusERP(epo, mnt, W, opt);

%% choose some interesting components
opt.selection= [1 3 7 9 14 16 17 21 25];
%% for printing
%opt.fontSize= 10;
plotPatternsPlusSpecERP(epo, mnt, W, opt);
