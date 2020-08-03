setup_augcog;
%nn= 4;
augcog = augcog(6:11);

fig_dir= 'augcog_misc/';
ival_audi_low= [[390 550]; [390 600];[350 500]; [350 600]; [250 400]; [450 580]];
ival_audi_high= [[350 650]; [420 650];[400 600]; [500 700]; [400 550]; [450 700]];
ival_visi_low= [[600 850]; [700 900];[550 850]; [800 900]; [500 700]; [800 1000]];
ival_visi_high= [[550 800]; [650 800];[600 950]; [750 900]; [550 750]; [750 1000]];
ival_comb_low= [[390 550]; [390 600];[350 500]; [400 700]; [250 400]; [420 600]];
ival_comb_high= [[350 650]; [420 650];[400 600]; [300 500]; [400 550]; [450 700]];

classDef = {{'I*','L*'},{'M*','R*'},'T*';'D','S','T'};
blk= getAugCogBlocks(augcog(nn).file);
blk= blk_selectBlocks(blk, {'*auditory','*visual','*comb'});
[cnt, bmrk, Mrk]= readBlocks(augcog(nn).file, blk, classDef);
bip= proc_bipolarChannels(cnt, 'Fp2-Eog', 'F7-F8', 'MT1-MT2');
bip.clab= {'EOGv','EOGh','EMG'};
cnt= proc_appendChannels(cnt, bip);
cnt= proc_removeChannels(cnt, 'Eog','MT1','MT2');
Mrk= mrk_addResponseLatency(Mrk, {{'D*','S*'},'T*'}, [0 1500]);
ie= getEventIndices(Mrk, 'D*');
ie2 = getEventIndices(Mrk,'S*');
valid= union(ie(find(~isnan(Mrk.latency(ie)))),ie2(find(isnan(Mrk.latency(ie2)))));
Mrk = mrk_selectEvents(Mrk,valid);
                    
%Mrk= mrk_selectClasses(Mrk, {'D*','S*'});

mnt= projectElectrodePositions(cnt.clab);
mnt.y= 1.2*mnt.y;
grd= sprintf('F3,FC1,Fz,FC2,F4,EOGv\nC3,CP1,Cz,CP2,C4,EOGh\nP3,O1,Pz,O2,P4,legend');
mnt= setDisplayMontage(mnt, grd);
ec= [chanind(mnt, 'EOG*'), length(mnt.clab)+1];
mnt.box(1,ec)= mnt.box(1,ec)+0.1;
scalp_opt= struct('shading','flat', 'resolution',20, 'contour',-5);
grid_opt= struct('colorOrder', [1 0 0; 0 0.7 0]);
grid_opt.scaleGroup= {scalpChannels, {'EOG*','Fp*'}, {'EMG'}, {'Ekg*'}};
grid_opt.scalePolicy= 'auto';
grid_opt.axisTitleFontWeight= 'bold';
grid_opt.axisTitleHorizontalAlignment= 'center';
rsqu_opt= struct('colorOrder',[0.8 0 0.7]);
rsqu_opt.axisTitleFontWeight= 'bold';
rsqu_opt.axisTitleHorizontalAlignment= 'center';
head= mnt;
head.x(chanind(head, 'Fp*','F7,8'))= NaN;

mrk= mrk_selectClasses(Mrk, '*auditory');
if length(mrk.pos)>0;
mrk= mrk_selectClasses(mrk, '*low*');

  
epo= makeEpochs(cnt, mrk, [-200 800]);
epo= proc_baseline(epo, [-200 0]);

grid_plot(epo, mnt, grid_opt);
valid_resp= find(~isnan(mrk.latency));
grid_markByBox(fractileValues(mrk.latency(valid_resp)));
ival= ival_audi_low(nn,:);
grid_markIval(ival);
saveFigure([fig_dir epo.title '_p300_auditory_low'], [12 5]*1.5);

epo_rsqu= proc_r_square(epo);
grid_plot(epo_rsqu, mnt, rsqu_opt);
saveFigure([fig_dir epo.title '_p300_rsqu_auditory_low'], [12 7]);

%plotClassTopographies(epo, mnt, ival, scalp_opt);
showERPscalps(epo, head, ival, 0, scalp_opt);
shiftAxesLeft; shiftAxesRight;
saveFigure([fig_dir epo.title '_p300_scalp_auditory_low'], [6 10]*2);

showERPscalps(epo, head, 200:100:800, 0, scalp_opt);
saveFigure([fig_dir epo.title '_p300_scalps_auditory_low'], [15 5]*1.5);


mrk= mrk_selectClasses(Mrk, '*auditory');
mrk= mrk_selectClasses(mrk, '*high*');
epo= makeEpochs(cnt, mrk, [-200 800]);
epo= proc_baseline(epo, [-200 0]);

grid_plot(epo, mnt, grid_opt);
valid_resp= find(~isnan(mrk.latency));
grid_markByBox(fractileValues(mrk.latency(valid_resp)));
ival= ival_audi_high(nn,:);
grid_markIval(ival);
saveFigure([fig_dir epo.title '_p300_auditory_high'], [12 5]*1.5);

epo_rsqu= proc_r_square(epo);
grid_plot(epo_rsqu, mnt, rsqu_opt);
saveFigure([fig_dir epo.title '_p300_rsqu_auditory_high'], [12 7]);

%plotClassTopographies(epo, mnt, ival, scalp_opt);
showERPscalps(epo, head, ival, 0, scalp_opt);
shiftAxesLeft; shiftAxesRight;
saveFigure([fig_dir epo.title '_p300_scalp_auditory_high'], [6 10]*2);

showERPscalps(epo, head, 200:100:800, 0, scalp_opt);
saveFigure([fig_dir epo.title '_p300_scalps_auditory_high'], [15 5]*1.5);

end


mrk= mrk_selectClasses(Mrk, '*comb');
if length(mrk.pos)>0;
mrk= mrk_selectClasses(mrk, '*low*');

  
epo= makeEpochs(cnt, mrk, [-200 800]);
epo= proc_baseline(epo, [-200 0]);

grid_plot(epo, mnt, grid_opt);
valid_resp= find(~isnan(mrk.latency));
grid_markByBox(fractileValues(mrk.latency(valid_resp)));
ival= ival_comb_low(nn,:);
grid_markIval(ival);
saveFigure([fig_dir epo.title '_p300_comb_low'], [12 5]*1.5);

epo_rsqu= proc_r_square(epo);
grid_plot(epo_rsqu, mnt, rsqu_opt);
saveFigure([fig_dir epo.title '_p300_rsqu_comb_low'], [12 7]);

%plotClassTopographies(epo, mnt, ival, scalp_opt);
showERPscalps(epo, head, ival, 0, scalp_opt);
shiftAxesLeft; shiftAxesRight;
saveFigure([fig_dir epo.title '_p300_scalp_comb_low'], [6 10]*2);

showERPscalps(epo, head, 200:100:800, 0, scalp_opt);
saveFigure([fig_dir epo.title '_p300_scalps_comb_low'], [15 5]*1.5);


mrk= mrk_selectClasses(Mrk, '*comb');
mrk= mrk_selectClasses(mrk, '*high*');
epo= makeEpochs(cnt, mrk, [-200 800]);
epo= proc_baseline(epo, [-200 0]);

grid_plot(epo, mnt, grid_opt);
valid_resp= find(~isnan(mrk.latency));
grid_markByBox(fractileValues(mrk.latency(valid_resp)));
ival= ival_comb_high(nn,:);
grid_markIval(ival);
saveFigure([fig_dir epo.title '_p300_comb_high'], [12 5]*1.5);

epo_rsqu= proc_r_square(epo);
grid_plot(epo_rsqu, mnt, rsqu_opt);
saveFigure([fig_dir epo.title '_p300_rsqu_comb_high'], [12 7]);

%plotClassTopographies(epo, mnt, ival, scalp_opt);
showERPscalps(epo, head, ival, 0, scalp_opt);
shiftAxesLeft; shiftAxesRight;
saveFigure([fig_dir epo.title '_p300_scalp_comb_high'], [6 10]*2);

showERPscalps(epo, head, 200:100:800, 0, scalp_opt);
saveFigure([fig_dir epo.title '_p300_scalps_comb_high'], [15 5]*1.5);

end


mrk= mrk_selectClasses(Mrk, '*visual');
if length(mrk.pos)>0
  mrk= mrk_selectClasses(mrk, '*low*');
epo= makeEpochs(cnt, mrk, [-200 1300]);
epo= proc_baseline(epo, [-200 0]);

grid_plot(epo, mnt, grid_opt);
valid_resp= find(~isnan(mrk.latency));
grid_markByBox(fractileValues(mrk.latency(valid_resp)));
ival= ival_visi_low(nn,:);
grid_markIval(ival);
saveFigure([fig_dir epo.title '_p300_visual_low'], [12 5]*1.5);

epo_rsqu= proc_r_square(epo);
grid_plot(epo_rsqu, mnt, rsqu_opt);
saveFigure([fig_dir epo.title '_p300_rsqu_visual_low'], [12 7]);

%plotClassTopographies(epo, mnt, ival, scalp_opt);
showERPscalps(epo, head, ival, 0, scalp_opt);
shiftAxesLeft; shiftAxesRight;
saveFigure([fig_dir epo.title '_p300_scalp_visual_low'], [6 10]*2);

showERPscalps(epo, head, 400:100:1000, 0, scalp_opt);
saveFigure([fig_dir epo.title '_p300_scalps_visual_low'], [15 5]*1.5);


mrk= mrk_selectClasses(Mrk, '*visual');
mrk= mrk_selectClasses(mrk, '*high*');
epo= makeEpochs(cnt, mrk, [-200 1300]);
epo= proc_baseline(epo, [-200 0]);

grid_plot(epo, mnt, grid_opt);
valid_resp= find(~isnan(mrk.latency));
grid_markByBox(fractileValues(mrk.latency(valid_resp)));
ival= ival_visi_high(nn,:);
grid_markIval(ival);
saveFigure([fig_dir epo.title '_p300_visual_high'], [12 5]*1.5);

epo_rsqu= proc_r_square(epo);
grid_plot(epo_rsqu, mnt, rsqu_opt);
saveFigure([fig_dir epo.title '_p300_rsqu_visual_high'], [12 7]);

%plotClassTopographies(epo, mnt, ival, scalp_opt);
showERPscalps(epo, head, ival, 0, scalp_opt);
shiftAxesLeft; shiftAxesRight;
saveFigure([fig_dir epo.title '_p300_scalp_visual_high'], [6 10]*2);

showERPscalps(epo, head, 400:100:1000, 0, scalp_opt);
saveFigure([fig_dir epo.title '_p300_scalps_visual_high'], [15 5]*1.5);



end


return




% if you want
crit.maxmin=100;
iArte= find_artifacts(epo, {'F3,z,4','C3,z,4','P3,z,4'}, crit);
fprintf('%d artifact trials removed (max-min>%d uV)\n', ...
        length(iArte), crit.maxmin);
epo= proc_selectEpochs(epo, setdiff(1:size(epo.x,3),iArte));

