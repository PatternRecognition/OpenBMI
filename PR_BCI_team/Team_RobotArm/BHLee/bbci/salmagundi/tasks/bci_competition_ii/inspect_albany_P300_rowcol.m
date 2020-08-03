file= 'bci_competition_ii/albany_P300_train';
fig_dir= 'bci_competition_ii/';

%[cnt, mrk, mnt]= loadProcessedEEG(file);
%Epo= makeEpochs(cnt, mrk, [-50 550]);
%Epo= proc_albanyAverageP300Trials(Epo, 15);
[Epo, mrk, mnt]= loadProcessedEEG(file, 'avg15');
mnt.x(64)= 0;
mnt.y(64)= [-0.6 1.6]*mnt.y([58 62]);
mnt.x(44)= [-0.6 1.6]*mnt.x([14 42]);
mnt.y(44)= 0;
mnt.x(43)= [-0.6 1.6]*mnt.x([8 41]);
mnt.y(43)= 0;

grd= sprintf('F7,F3,Fz,F4,legend\nFC5,FC3,FCz,FC4,FC6\nC5,C3,Cz,C4,C6\nPO7,P3,Pz,P4,PO8');
mnt= setDisplayMontage(mnt, grd);

rsqu_opt= struct('yUnit','', 'colorOrder',[1 0 0]);
scalp_opt= struct('colAx','range', 'contour',-4);


epo= proc_selectEpochs(Epo, find(Epo.code>6));
epo.className= {'correct row', 'wrong row'};


epo= proc_baseline(epo, [-50 0]);
grid_plot(epo, mnt);
saveFigure([fig_dir 'albany_P300_row2_erp'], [10 6]*2);


epo_rsqu= proc_r_square(epo);
grid_plot(epo_rsqu, mnt, rsqu_opt);
grid_markIval([290 350]);
saveFigure([fig_dir 'albany_P300_row2_rsqu'], [10 6]*2);

clf;
topo= proc_meanAcrossTime(epo_rsqu, [290 350]);
plotScalpPattern(mnt, topo.x, scalp_opt);
title(sprintf('%s  [290 350] ms', topo.className{1}));
saveFigure([fig_dir 'albany_P300_row2_rsqu_topo'], [6 5]*2);




epo= proc_selectEpochs(Epo, find(Epo.code<=6));
epo.className= {'correct col', 'wrong col'};


epo= proc_baseline(epo, [-50 0]);
grid_plot(epo, mnt);
saveFigure([fig_dir 'albany_P300_col2_erp'], [10 6]*2);


epo_rsqu= proc_r_square(epo);
grid_plot(epo_rsqu, mnt, rsqu_opt);
grid_markIval([270 350]);
saveFigure([fig_dir 'albany_P300_col2_rsqu'], [10 6]*2);

clf;
topo= proc_meanAcrossTime(epo_rsqu, [270 350]);
plotScalpPattern(mnt, topo.x, scalp_opt);
title(sprintf('%s  [270 350] ms', topo.className{1}));
saveFigure([fig_dir 'albany_P300_col2_rsqu_topo'], [6 5]*2);



epo= Epo;
epo.y= [Epo.code<=6 & Epo.y(1,:); ...
        Epo.code>6  & Epo.y(1,:); ...
        Epo.code<=6 & Epo.y(2,:);
        Epo.code>6  & Epo.y(2,:)],
epo.className= {'correct col', 'correct row', ...
                'wrong col', 'wrong row'};

opt= struct('colorOrder', [1 0 0;  0.85 0.85 0;  0 0.7 0; 0 0.8 0.7]);
epo= proc_baseline(epo, [-50 0]);
grid_plot(epo, mnt, opt);
saveFigure([fig_dir 'albany_P300_rowcol2_erp'], [10 6]*2);
