file= 'Guido_05_11_15/imag_1drfbGuido';
%file= 'VPcp_05_10_28/imag_1drfbVPcp';
grd= sprintf('EOGh,F3,Fz,F4,EOGv\nC3,C1,Cz,C2,C4\nCP3,CP1,CPz,CP2,CP4\nlegend,P3,Pz,P4,scale');

[cnt,mrk,mnt]= eegfile_loadMatlab(file);
cnt= proc_selectChannels(cnt, 'not','E*');
mnt= mnt_setGrid(mnt, grd);
mnt= mnt_excenterNonEEGchans(mnt);
mnt.x= 1.2*mnt.x;
mnt.y= 1.2*mnt.y;

mrk_err= copy_struct(mrk, 'fs');
mrk_err.pos= mrk.pos+mrk.duration/1000*mrk.fs;
mrk_err.y= [~mrk.ishit; mrk.ishit]; 
mrk_err.className= {'miss','hit'};

cnt_flt= proc_subtractMovingAverage(cnt, 250);
[mrk_err, rClab]= reject_varEventsAndChannels(cnt_flt, mrk_err, [-250 750], ...
                                              'do_bandpass', 0);
clab_rm= rClab;
mnt_red= mnt_restrictMontage(mnt, 'not', clab_rm);
clear cnt_flt
epo= makeEpochs(cnt, mrk_err, [-250 750]);
epo= proc_baseline(epo, [-250 0]);
epo_r= proc_r_square_signed(epo);
epo_r.x= 100*epo_r.x;

%H= grid_plot(epo, mnt, opt_grid);
scalpEvolutionPlusChannel(epo, mnt_red, 'FCz', ival_scalps, opt_scalp);


clf;
erp= proc_selectChannels(epo, 'FCz');
erp_r= proc_r_square_signed(erp);
subplot(1,2,1);
plotChannel(erp, 1)
subplot(1,2,2);
plotChannel(erp_r, 1)


ep= proc_selectIval(epo, [200 300]);
ep= proc_meanAcrossTime(ep);
ff= proc_flaten(ep);
C= trainClassifier(ff, {'RLDA', 0.01});

clf;
subplot(1,3,1);
scalpPlot(mnt, C.w);
subplot(1,3,2);
erp= proc_linearDerivation(epo, C.w);
erp_r= proc_r_square_signed(erp);
plotChannel(erp, 1)
subplot(1,3,3);
plotChannel(erp_r, 1)



ep= proc_selectIval(epo, [200 300]);
ep= proc_meanAcrossTime(ep);
ep= proc_selectClasses(ep, 'miss');
ep= proc_average(ep);
A= ep.x;
S= cov(cnt.x);
W= pinv(S)*A';

clf;
subplot(1,2,1);
scalpPlot(mnt, A);
subplot(1,2,2);
scalpPlot(mnt, W);


clf;
erp= proc_linearDerivation(epo, W);
erp_r= proc_r_square_signed(erp);
subplot(1,2,1);
plotChannel(erp, 1)
subplot(1,2,2);
plotChannel(erp_r, 1)




ep= proc_selectIval(epo, [200 300]);
ep= proc_meanAcrossTime(ep);
ep= proc_r_square_signed(ep);
A= ep.x;
S= cov(cnt.x);
W= pinv(S)*A';

clf;
subplot(1,2,1);
scalpPlot(mnt, A);
subplot(1,2,2);
scalpPlot(mnt, W);


clf;
erp= proc_linearDerivation(epo, W);
erp_r= proc_r_square_signed(erp);
subplot(1,2,1);
plotChannel(erp, 1)
subplot(1,2,2);
plotChannel(erp_r, 1)


