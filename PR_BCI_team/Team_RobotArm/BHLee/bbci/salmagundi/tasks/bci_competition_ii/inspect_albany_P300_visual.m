file= 'bci_competition_ii/albany_P300_train';
fig_dir= 'bci_competition_ii/';
xTrials= [10 10];

[cnt, mrk, mnt]= loadProcessedEEG(file);
%cnt= proc_selectChannels(cnt, 'T#','CP7,8','P#','PO#','O#','Iz');
mnt.x(64)= 0;
mnt.y(64)= [-0.6 1.6]*mnt.y([58 62]);
mnt.x(44)= [-0.6 1.6]*mnt.x([14 42]);
mnt.y(44)= 0;
mnt.x(43)= [-0.6 1.6]*mnt.x([8 41]);
mnt.y(43)= 0;


grd= sprintf('T7,P7,legend,P8,T8\nP3,P1,Pz,P2,P4\nP5,PO3,POz,PO4,P6\nPO7,O1,Oz,O2,PO8');
mnt_visual= setDisplayMontage(mnt, grd);

rsqu_opt= struct('yUnit','', 'colorOrder',[1 0 0]);
scalp_opt= struct('colAx','range', 'contour',-5);


mrk_row= mrk_selectEvents(mrk, mrk.code>6);
mrk_row.className= {'correct row', 'wrong row'};

epo= makeEpochs(cnt, mrk_row, [-50 550]);
epo= proc_baseline(epo, [-50 0]);
grid_plot(epo, mnt_visual);
saveFigure([fig_dir 'albany_P300_row_erp'], [10 6]*2);

epo_rsqu= proc_r_square(epo);
grid_plot(epo_rsqu, mnt_visual, rsqu_opt);
grid_markIval([240 260]);
grid_markIval([450 470]);
saveFigure([fig_dir 'albany_P300_row_rsqu'], [10 6]*2);

clf;
topo= proc_meanAcrossTime(epo_rsqu, [240 260]);
plotScalpPattern(mnt, topo.x, scalp_opt);
title(sprintf('%s  [240 260] ms', topo.className{1}));
saveFigure([fig_dir 'albany_P300_row_rsqu_topo'], [6 5]*2);

clf;
topo= proc_meanAcrossTime(epo_rsqu, [450 470]);
plotScalpPattern(mnt, topo.x, scalp_opt);
title(sprintf('%s  [450 470] ms', topo.className{1}));
saveFigure([fig_dir 'albany_P300_row_rsqu_450_470_topo'], [6 5]*2);

fv= proc_commonAverageReference(epo);
fv= proc_baseline(fv, [150 200]);
fv= proc_selectIval(fv, [220 450]);
fv= proc_jumpingMeans(fv, 5);
colormap(hot(21));
fv= rmfield(fv, 'title');
plot_classification_scalp(fv, mnt, 'LDA');
title(sprintf('<%s> vs. <%s>  [220 450] ms', fv.className{:}));
saveFigure([fig_dir 'albany_P300_row_classy_scalp'], [6 5]*2);


% $$$ fv= proc_baseline(epo, [150 200]);
% $$$ fv= proc_selectIval(fv, [240 260]);
% $$$ fv= proc_meanAcrossTime(fv);
% $$$ model= struct('classy', 'LPM');
% $$$ model.param= [0.1 1 10 100];
% $$$ classy= selectModel(fv, model, [3 10]);
% $$$ C= trainClassifier(fv, classy);
% $$$ clf;
% $$$ colormap(green_white_red(11,0.9));
% $$$ plotScalpPattern(mnt, C.w);
% $$$ saveFigure([fig_dir 'albany_P300_row_weighted_scalp'], [7 5]*2);
% $$$ colormap('default');


fv= proc_selectChannels(epo, 'T8,10','CP7,8','P#','PO#','O#','Iz');
fv= proc_baseline(fv, [0 150]);
fv= proc_selectIval(fv, [220 450]);
fv= proc_jumpingMeans(fv, 5);
doXvalidationPlus(fv, 'LDA', xTrials, 3);
%% 15.6±0.2%  (fn: 18.3±0.5%,  fp: 14.6±0.2%)  [train: 12.8±0.0%]




mrk_col= mrk_selectEvents(mrk, mrk.code<=6);
mrk_col.className= {'correct col', 'wrong col'};

epo= makeEpochs(cnt, mrk_col, [-50 550]);
epo= proc_baseline(epo, [-50 0]);
grid_plot(epo, mnt_visual);
saveFigure([fig_dir 'albany_P300_col_erp'], [10 6]*2);

epo_rsqu= proc_r_square(epo);
grid_plot(epo_rsqu, mnt_visual, rsqu_opt);
grid_markIval([250 280]);
grid_markIval([420 470]);
saveFigure([fig_dir 'albany_P300_col_rsqu'], [10 6]*2);

clf;
topo= proc_meanAcrossTime(epo_rsqu, [250 280]);
plotScalpPattern(mnt, topo.x, scalp_opt);
title(sprintf('%s  [250 280] ms', topo.className{1}));
saveFigure([fig_dir 'albany_P300_col_rsqu_topo'], [6 5]*2);

clf;
topo= proc_meanAcrossTime(epo_rsqu, [420 470]);
plotScalpPattern(mnt, topo.x, scalp_opt);
title(sprintf('%s  [420 470] ms', topo.className{1}));
saveFigure([fig_dir 'albany_P300_col_rsqu_420_470_topo'], [6 5]*2);

fv= proc_commonAverageReference(epo);
fv= proc_baseline(fv, [150 200]);
fv= proc_selectIval(fv, [220 450]);
fv= proc_jumpingMeans(fv, 5);
fv= rmfield(fv, 'title');
colormap(hot(21));
plot_classification_scalp(fv, mnt, 'LDA');
title(sprintf('<%s> vs. <%s>  [220 450] ms', fv.className{:}));
saveFigure([fig_dir 'albany_P300_col_classy_scalp'], [6 5]*2);

fv= proc_commonAverageReference(epo);
fv= proc_baseline(fv, [150 200]);
fv= proc_selectIval(fv, [400 550]);
fv= proc_jumpingMeans(fv, 6);
fv= rmfield(fv, 'title');
colormap(hot(21));
plot_classification_scalp(fv, mnt, 'LDA');
title(sprintf('<%s> vs. <%s>  [400 550] ms', fv.className{:}));
saveFigure([fig_dir 'albany_P300_col_400_550_classification_scalp'], [6 5]*2);

fv= proc_selectChannels(epo, 'P7-3','P4-8','PO#','O#','Iz');
fv= proc_baseline(fv, [150 200]);
fv= proc_selectIval(fv, [220 450]);
fv= proc_jumpingMeans(fv, 5);
doXvalidationPlus(fv, 'LDA', xTrials, 3);
%% 12.6±0.1%  (fn: 15.3±0.4%,  fp: 12.1±0.2%)  [train: 10.8±0.0%]

%% this could provide additional information
fv= proc_selectChannels(epo, 'P7-3','P4-8','PO#','O#','Iz');
fv= proc_baseline(fv, [0 150]);
fv= proc_selectIval(fv, [400 550]);
fv= proc_jumpingMeans(fv, 6);
doXvalidationPlus(fv, 'LDA', xTrials, 3);
%% 31.3±0.4%  (fn: 32.8±0.3%,  fp: 31.3±0.4%)  [train: 29.1±0.1%]



mrk4= mrk;
mrk4.y= [mrk.code<=6 & mrk.y(1,:); ...
         mrk.code>6 & mrk.y(1,:); ...
         mrk.code<=6 & mrk.y(2,:);
         mrk.code>6 & mrk.y(2,:)],
mrk4.className= {'correct col', 'correct row', ...
                 'wrong col', 'wrong row'};

opt= struct('colorOrder', [1 0 0;  0.85 0.85 0;  0 0.7 0; 0 0.8 0.7]);
epo= makeEpochs(cnt, mrk4, [-50 550]);
epo= proc_baseline(epo, [-50 0]);
grid_plot(epo, mnt_visual, opt);
saveFigure([fig_dir 'albany_P300_col_row_erp'], [10 6]*2);
