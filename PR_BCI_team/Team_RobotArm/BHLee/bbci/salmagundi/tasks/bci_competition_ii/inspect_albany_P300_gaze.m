file= 'bci_competition_ii/albany_P300_train';
fig_dir= 'bci_competition_ii/';
xTrials= [10 10];

[cnt, mrk, mnt]= loadProcessedEEG(file);
mnt.x(64)= 0;
mnt.y(64)= [-1 2]*mnt.y([58 62]);
mnt.x(44)= [-1 2]*mnt.x([14 42]);
mnt.y(44)= 0;
mnt.x(43)= [-1 2]*mnt.x([8 41]);
mnt.y(43)= 0;


grd= sprintf('T7,P7,legend,P8,T8\nP3,P1,Pz,P2,P4\nP5,PO3,POz,PO4,P6\nPO7,O1,Oz,O2,PO8');
mnt_visual= setDisplayMontage(mnt, grd);

rsqu_opt.yUnit= '';
rsqu_opt.colorOrder= [1 0 0];



lett_matrix= reshape(['A':'Z', '1':'9', ' '], 6, 6)'; 
nTrials= length(mrk.code);
nHighPerLett= 15*12;
nLetts= nTrials/nHighPerLett;
lett= char(zeros(1, nLetts));
target= zeros(nLetts,2);
for il= 1:nLetts,
  iv= [1:12] + (il-1)*nHighPerLett;
  ir= find(mrk.toe(iv)==1);
  target(il,:)= sort(mrk.code(iv(ir))) - [0 6];
  lett(il)= lett_matrix(target(il,2), target(il,1));
end
lett


Target= target(repmat(1:nLetts, [nHighPerLett 1]), :);
stimOverGaze= ( ismember(mrk.code, 7) & ismember(Target(:,2)', 3:6) ) | ...
              ( ismember(mrk.code, 8) & ismember(Target(:,2)', 4:6) ) | ...
              ( ismember(mrk.code, 9) & ismember(Target(:,2)', 5:6) ) | ...
              ( ismember(mrk.code, 10) & ismember(Target(:,2)', 6) );
stimUnderGaze= ( ismember(mrk.code, 12) & ismember(Target(:,2)', 1:4) ) | ...
               ( ismember(mrk.code, 11) & ismember(Target(:,2)', 1:3) ) | ...
               ( ismember(mrk.code, 10) & ismember(Target(:,2)', 1:2) ) | ...
               ( ismember(mrk.code, 9) & ismember(Target(:,2)', 1) );
mrk_gaze= mrk;
mrk_gaze.y= [stimOverGaze; stimUnderGaze];
mrk_gaze.className= {'stim over', 'stim under'};
mrk_gaze= mrk_selectEvents(mrk_gaze, find(any(mrk_gaze.y)));

epo= makeEpochs(cnt, mrk_gaze, [-50 500]);
epo= proc_baseline(epo, [-50 0]);
grid_plot(epo, mnt_visual);
saveFigure([fig_dir 'albany_P300_gaze_erp'], [10 7]*2);

epo_rsqu= proc_r_square(epo);
grid_plot(epo_rsqu, mnt_visual, rsqu_opt);
grid_markIval([240 270]);
grid_markIval([385 425]);
saveFigure([fig_dir 'albany_P300_gaze_rsqu'], [10 7]*2);


clf;
topo= proc_meanAcrossTime(epo_rsqu, [240 270]);
scalp_opt= struct('colAx','range', 'contour',0.005);
plotScalpPattern(mnt, topo.x, scalp_opt);
title(sprintf('%s  [%d %d] ms', topo.className{1}, [240 270]));
saveFigure([fig_dir 'albany_P300_gaze_rsqu_topo'], [6 5]*2);

clf;
topo= proc_meanAcrossTime(epo_rsqu, [385 425]);
scalp_opt= struct('colAx', 'range', 'contour',0.005);
plotScalpPattern(mnt, topo.x, scalp_opt);
title(sprintf('%s  [%d %d] ms', topo.className{1}, [385 425]));
saveFigure([fig_dir 'albany_P300_gaze_rsqu_385_425_topo'], [6 5]*2);


fv= proc_selectChannels(epo, 'T9,7','FT7','CP3-4','P5-6','PO#','O#','Iz');
fv= proc_baseline(fv, [150 200]);
fv= proc_selectIval(fv, [220 450]);
fv= proc_jumpingMeans(fv, 5);
doXvalidationPlus(fv, 'LDA', xTrials, 3);
%% 12.1±0.4%  (fn: 13.4±0.8%,  fp: 11.8±0.5%)  [train: 6.8±0.0%]

model= struct('classy','RLDA', 'msDepth',2, 'inflvar',2);
model.param= [0 0.001 0.01 0.1];
classy= selectModel(fv, model, [3 10 round(9/10*sum(any(fv.y)))]);
doXvalidationPlus(fv, classy, xTrials, 3);
%% 11.1±0.2%  (fn: 13.1±0.6%,  fp: 10.5±0.3%)  [train: 7.0±0.0%]

fv= proc_commonAverageReference(epo);
fv= proc_baseline(fv, [150 200]);
fv= proc_selectIval(fv, [220 450]);
fv= proc_jumpingMeans(fv, 5);

clf;
colormap(hot(21));
fv= rmfield(fv, 'title');
plot_classification_scalp(fv, mnt, 'LDA');
saveFigure([fig_dir 'albany_P300_gaze_classy_scalp'], [6 5]*2);
title(sprintf('<%s> vs. <%s>  [220 450] ms', fv.className{:}));
colormap default



mrk_gaze= mrk;
stimAtGaze= (mrk.code-6==Target(:,2)');
mrk_gaze.y= [stimOverGaze; stimUnderGaze; stimAtGaze];
mrk_gaze.className= {'stim over', 'stim under', 'correct row'};
mrk_gaze= mrk_selectEvents(mrk_gaze, find(any(mrk_gaze.y)));


epo= makeEpochs(cnt, mrk_gaze, [-50 500]);
epo= proc_baseline(epo, [-50 0]);
grid_plot(epo, mnt_visual);
saveFigure([fig_dir 'albany_P300_gaze_all_erp'], [10 7]*2);

fv= proc_selectChannels(epo, 'T9,7','FT7','CP3-4','P5-6','PO#','O#','Iz');
fv= proc_baseline(fv, [150 200]);
fv= proc_selectIval(fv, [220 450]);
fv= proc_jumpingMeans(fv, 5);
classy= {'constraint', 'LDA'};
doXvalidationPlus(fv, classy, xTrials);
%% 17.3±0.2%, [train: 10.8±0.1%]

%% TODO: constraint RLDA


Target= target(repmat(1:nLetts, [nHighPerLett 1]), :);
stimLeftOfGaze= ( ismember(mrk.code, 1) & ismember(Target(:,1)', 3:6) ) | ...
                ( ismember(mrk.code, 2) & ismember(Target(:,1)', 4:6) ) | ...
                ( ismember(mrk.code, 3) & ismember(Target(:,1)', 5:6) ) | ...
                ( ismember(mrk.code, 4) & ismember(Target(:,1)', 6) );
stimRightOfGaze= ( ismember(mrk.code, 6) & ismember(Target(:,1)', 1:4) ) | ...
                 ( ismember(mrk.code, 5) & ismember(Target(:,1)', 1:3) ) | ...
                 ( ismember(mrk.code, 4) & ismember(Target(:,1)', 1:2) ) | ...
                 ( ismember(mrk.code, 3) & ismember(Target(:,1)', 1) );
mrk_gaze= mrk;
mrk_gaze.y= [stimLeftOfGaze; stimRightOfGaze];
mrk_gaze.className= {'stim left', 'stim right'};
mrk_gaze= mrk_selectEvents(mrk_gaze, find(any(mrk_gaze.y)));

epo= makeEpochs(cnt, mrk_gaze, [-50 500]);
epo= proc_baseline(epo, [-50 0]);
grid_plot(epo, mnt_visual);
saveFigure([fig_dir 'albany_P300_gaze_lr_erp'], [10 7]*2);

epo_rsqu= proc_r_square(epo);
rsqu_opt.yUnit= '';
rsqu_opt.colorOrder= [1 0 0];
grid_plot(epo_rsqu, mnt_visual, rsqu_opt);
grid_markIval([80 100]);
saveFigure([fig_dir 'albany_P300_gaze_lr_rsqu'], [10 7]*2);

clf;
topo= proc_meanAcrossTime(epo_rsqu, [80 100]);
scalp_opt= struct('colAx','range', 'contour',-5);
plotScalpPattern(mnt, topo.x, scalp_opt);
title(sprintf('%s  [%d %d] ms', topo.className{1}, [80 100]));
saveFigure([fig_dir 'albany_P300_gaze_lr_rsqu_topo'], [6 5]*2);


fv= proc_commonAverageReference(epo);
fv= proc_baseline(fv, [-50 0]);
fv= proc_selectIval(fv, [50 150]);
fv= proc_jumpingMeans(fv, 3);

clf;
colormap(hot(21));
fv= rmfield(fv, 'title');
plot_classification_scalp(fv, mnt, 'LDA');
saveFigure([fig_dir 'albany_P300_gaze_lr_classy_scalp'], [6 5]*2);


fv= proc_selectChannels(epo, 'T9','C7,2,4','CP2-6','P2-8','POz,4,8','Oz','O2');
fv= proc_baseline(fv, [-50 0]);
fv= proc_selectIval(fv, [50 150]);
fv= proc_jumpingMeans(fv, 3);
doXvalidationPlus(fv, 'LDA', [3 10], 3);
%% >40%

fv= proc_selectChannels(epo, 'C7,5,4,6', 'CP7,5,3,4,6,8', ...
                             'P7,5,3,4,6,8', 'PO7,8', 'O1,2');
fv= proc_baseline(fv, [150 200]);
fv= proc_selectIval(fv, [220 450]);
fv= proc_jumpingMeans(fv, 5);
doXvalidationPlus(fv, 'LDA', [3 10], 3);
%% 28.3±0.5%  (fn: 31.1±1.0%,  fp: 26.4±0.6%)  [train: 22.1±0.1%]

model= struct('classy','RLDA', 'msDepth',2, 'inflvar',2);
model.param= [0 0.001 0.01 0.1];
classy= selectModel(fv, model, [3 10 round(9/10*sum(any(fv.y)))]);
doXvalidationPlus(fv, classy, xTrials, 3);
