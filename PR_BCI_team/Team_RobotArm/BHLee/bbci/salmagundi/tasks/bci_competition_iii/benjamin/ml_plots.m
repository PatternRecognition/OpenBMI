su= 'A';
%su= 'B';

model_RLDA= struct('classy','RLDA', 'msDepth',2, 'inflvar',1);
model_RLDA.param= [0 0.01 0.1 0.3];

map1= cmap_hsv_fade(10, 1/6, [0 1], 1);
map2= cmap_hsv_fade(11, [1/6 0], 1, 1);
cmap_fire= [map1; map2(2:end,:)];

fig_pre= ['preliminary/bci_competition_iii/benjamin/data_set_ii_' su '_'];

[epo,mrk,mnt]= ...
    loadProcessedEEG([EEG_IMPORT_DIR 'bci_competition_iii/data_set_ii_' su],...
                     'ave15');

fv= proc_commonAverageReference(epo);
fv= proc_baseline(fv, [0 150]);
fv= proc_selectIval(fv, [200 500]);
fv= proc_jumpingMeans(fv, 5);

opt_xv= struct('xTrials',[5 5], ...
               'outer_ms', 1, ...
               'msTrials',[3 5 -1], ...
               'loss',{{'classwiseNormalized', sum(fv.y,2)}});

colormap(flipud(cmap_fire));
plot_classification_scalp(fv, mnt, 'title','', 'opt_xv',opt_xv);
saveFigure([fig_pre 'erp_classy_scalp'], [12 10]);

plot_incremental_channel_selection(fv, mnt, 'nSelect', 12, ...
                                   'model',model_RLDA, 'opt_xv',opt_xv);
saveFigure([fig_pre 'erp_incremental_selection'], [20 15]);

%% linear programming machine
model_LPM= struct('classy','LPM', 'std_factor',2);
model_LPM.param= struct('index',2, 'scale','log', ...
                        'value', [-2:2:4]);
classy= select_model(fv, model_LPM, opt_xv);
C= trainClassifier(fv, classy);

clab_replace= {'Fpz','Fp'; 'FAF1','FAF';
               'Fz','F'; 'FFC1','FFC'; 'FCz','FC'; 'CFC1','CFC';
               'Cz','C'; 'CCP1','CCP'; 'CPz','CP'; 'PCP1','PCP';
               'Pz','P'; 'PPO1','PPO'; 'POz','PO'; 'Oz','O'};
ii= chanind(fv, 'not',clab_replace{:,1});
ic= chanind(fv, clab_replace{:,1});
is= strpatternmatch('*1', clab_replace(:,1));
%fv.clab(ii)= '';
yt= ic;
yt(is)= yt(is)+0.5;
[so,si]= sort(yt);

hnd= plot_classifierImage(C, fv, 'show_title',0, 'fontSize',10);
set(hnd.ax(1), 'yTick',yt(si), 'yTickLabel',clab_replace(si,2));
saveFigure([fig_pre 'erp_sparse_classy'], [20 12]);
