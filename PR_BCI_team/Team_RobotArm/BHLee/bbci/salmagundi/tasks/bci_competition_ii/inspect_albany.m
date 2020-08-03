%subject= 'AA'; dscr_band= [11 14; 23 27]
%subject= 'BB'; dscr_band= [12 15; 25 28]
%subject= 'CC'; dscr_band= [10 14; 23 26]
sub_dir= 'bci_competition_ii/';
if ~exist('subject','var'), error('please define subject and dscr_band'); end 
  
file= sprintf('%salbany_%s_train', sub_dir, subject);
fig_root= [sub_dir 'albany_' subject '_'];

[cnt, Mrk, mnt]= loadProcessedEEG(file);
mrk= mrk_selectClasses(Mrk, {'top','bottom'});

epo= makeEpochs(cnt, mrk, [-1000 4000]);

mnt_spec= mnt;
mnt_spec.box_sz= 0.9*mnt_spec.box_sz;

grid_opt4= struct('colorOrder', [1 0 0; 0.9 0.8 0; 0.8 0 0.8; 0 0 1]);

spec_opt= struct('yUnit','power', 'xTickMode','auto');
spec_opt.colorOrder= [1 0 0; 0 0 1];

spec= proc_selectIval(epo, [500 4000]);
spec= proc_commonAverageReference(spec);
spec= proc_spectrum(spec, [4 35], epo.fs);
grid_plot(spec, mnt_spec, spec_opt);
saveFigure([fig_root 'spec'], [10 8]*2);

epo_rsqu= proc_r_square(spec);
rsqu_opt= spec_opt;
rsqu_opt.yUnit= '';
rsqu_opt.colorOrder= [1 0 0];
grid_plot(epo_rsqu, mnt_spec, rsqu_opt);
for ib= 1:size(dscr_band,1),
  grid_markIval(dscr_band(ib,:));
end
saveFigure([fig_root 'spec_rsqu'], [10 8]*2);


clear spec epo


for ib= 1:size(dscr_band,1),

band= dscr_band(ib,:);
clf;
topo= proc_meanAcrossTime(epo_rsqu, band);
scalp_opt= struct('contour',-5, 'colAx','range');
plotScalpPattern(mnt, topo.x(:,:,1), scalp_opt);
title(sprintf('%s  [%d-%d Hz]', topo.className{1}, band));
saveFigure(sprintf('%sspec_rsqu_%d_%d_topo', fig_root, trunc(band)), [6 5]*2);


clear cnt_flt epo_flt
[cnt, Mrk, mnt]= loadProcessedEEG(file);
mrk= mrk_selectClasses(Mrk, {'top','bottom'});

band= dscr_band(ib,:) + [-1 1];
[b,a]= getButterFixedOrder(band, cnt.fs, 6);
cnt_flt= proc_filt(cnt, b, a);
cnt_flt.title= sprintf('%s [%d-%d Hz]', cnt.title, band);
clear cnt

epo_flt= makeEpochs(cnt_flt, mrk, [-1000 4000]);

csp_ival= [1000 3000];
csp= proc_selectIval(epo_flt, csp_ival);
[csp, csp_w, la]= proc_csp(csp, 2);
head= setDisplayMontage(mnt, 'visible_128');
head= restrictDisplayChannels(mnt, epo_flt.clab);
opt= struct('shading','flat', 'resolution',80, 'colAx','sym');
plotCSPatterns(csp, head, csp_w, la, opt);
saveFigure(sprintf('%s%g_%g_csp', fig_root, trunc(band)), [10 7]*1.5);

erd_refIval= [-1000 0];
epo_flt= makeEpochs(cnt_flt, Mrk, [-1000 4000]);
erd= proc_linearDerivation(epo_flt, csp_w);
erd.clab= csp.clab;
erd= proc_squareChannels(erd);
erd= proc_average(erd);
erd= proc_calcERD(erd, erd_refIval, 150);
grd= sprintf('top:csp1,_,top:csp2\nbottom:csp1,legend,bottom:csp2');
csp_mnt= setElectrodeMontage(erd.clab, grd);
grid_plot(erd, csp_mnt, grid_opt4);
grid_markIval(csp_ival);
saveFigure(sprintf('%s%g_%g_csp_erd', fig_root, trunc(band)), [9 8]*1.5);

clear csp

erd= proc_commonAverageReference(epo_flt);
erd= proc_selectChannels(erd, 'C5,3,z,4,6', 'CP5,3,z,4,6', 'P5,6', 'Oz');
erd= proc_squareChannels(erd);
erd= proc_average(erd);
erd= proc_calcERD(erd, erd_refIval, 150);
grd= sprintf('C5,C3,Cz,C4,C6\nCP5,CP3,CPz,CP4,CP6\nP5,legend,Oz,_,P6');
erd_mnt= setDisplayMontage(mnt, grd);
grid_plot(erd, erd_mnt, grid_opt4);
saveFigure(sprintf('%s%g_%g_erd', fig_root, trunc(band)), [10 7]*1.5);

end  %% for ib



return



clear cnt_flt epo_flt
[cnt, Mrk, mnt]= loadProcessedEEG(file);
mrk= mrk_selectClasses(Mrk, {'top','bottom'});

erd= proc_selectChannels(cnt, 'C5-6','CP5-6','P5,6','Oz');
[erd,erd]= phaseAmplitudeEstimate(erd, 12);
erd= makeEpochs(erd, Mrk, [-1000 4000]);
grid_plot(erd, erd_mnt);


%epo= proc_baseline(epo, [-1000 0]);
%grid_plot(epo, mnt);
%saveFigure([fig_root '_erp'], [10 8]*2);
