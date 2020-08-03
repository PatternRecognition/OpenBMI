file= 'Klaus_03_11_04/imagKlaus';

%% options for plotting
opt= struct('colorOrder', [1 0 0; 0 0.7 0; 0 0 1; 0.9 0.8 0]);
opt.xGrid= 'on';
opt.yZeroLine= 'on';
spec_opt= opt;
spec_opt.xTick= 10:10:40;
spec_opt.xTickLabel= '';
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

epo= makeEpochs(cnt_active, mrk_active, [0 3500]);
head= restrictMontage(mnt, epo.clab);
plotPatternsPlusSpec(epo, head, W, opt);

%% choose some interesting components
opt.selection= [ ... ];
%% for printing
%opt.fontSize= 10;
plotPatternsPlusSpec(epo, mnt, W, opt);
