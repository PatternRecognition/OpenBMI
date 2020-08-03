sub_dir= 'bci_competition_ii/';
subject= 'AA'; dscr_band= [10 13.5];
%subject= 'BB'; dscr_band= [11 15];
%subject= 'CC'; dscr_band= [10 14];
xTrials= [5 10];

file= sprintf('%salbany_%s_train', sub_dir, subject);
fig_root= [sub_dir 'albany_' subject '_'];

[cnt, mrk, mnt]= loadProcessedEEG(file);

mrk= pickEvents(mrk, {'top','bottom'});
epo= makeEpochs(cnt, mrk, [-1000 4500]);


fv= proc_selectIval(epo, [1000 3000]);
fv= proc_commonAverageReference(fv);
fv= proc_fourierBandMagnitude(fv, band, 320);
fv= proc_jumpingMeans(fv, 2);

clf;
colormap(hot(21));
plot_classification_scalp(fv, mnt, 'LDA');
saveFigure([fig_root 'classification_scalp'], [9 8]*2);


band= dscr_band + [-1 1];
csp_ival= [1000 3000];
[b,a]= getButterFixedOrder(band, epo.fs, 6);
epo_flt= proc_filtfilt(epo, b, a);
fv= proc_selectIval(epo_flt, csp_ival);
[fv, csp_w]= proc_csp(fv, 2);
fv= proc_variance(fv);
doXvalidation(fv, 'LDA', xTrials);

scalp_opt= struct('shading','flat', 'resolution',24);
for ic= 1:4,
  subplot(2,2,ic);
  plotScalpPattern(mnt, csp_w(:,ic), scalp_opt);
end
saveFigure([fig_root 'csp_patterns'], [9 8]*2);



csp= proc_linearDerivation(epo_flt, csp_w);
csp.clab= {'csp1','csp3','csp4', 'csp2'};
erd= proc_selectIval(csp, [-1000 4500]);
erd.title= sprintf('%s [%d %d] Hz', erd.title, band);
erd= proc_squareChannels(erd);
erd= proc_average(erd);
erd= proc_calcERD(erd, [-1000 0], 100);

grd= sprintf('csp1,_,csp2\ncsp3,legend,csp4');
csp_mnt= setElectrodeMontage(csp.clab, grd);
grid_plot(erd, csp_mnt);
grid_markIval(csp_ival);
saveFigure([fig_root 'csp_erd'], [9 8]*2);
