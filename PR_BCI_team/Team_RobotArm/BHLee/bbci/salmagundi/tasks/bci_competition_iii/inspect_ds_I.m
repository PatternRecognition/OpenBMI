file= [EEG_IMPORT_DIR 'bci_competition_iii/tuebingen/Competition_train'];
S= load(file);
epo= struct('x', permute(S.X, [3 2 1]));
epo.y= double([S.Y'==-1; S.Y'==1]);
epo.className= {'left','tongue'};
epo.fs= 1000;
epo.t= 501:3500;
clear mnt S
for ii= 1:8,
  for jj= 1:8,
    idx= ii+(jj-1)*8;
    mnt.clab{idx}= sprintf('ch_%d_%d', ii, jj);
    mnt.x(idx,1)= (ii-4.5)/5;
    mnt.y(idx,1)= (4.5-jj)/5;
    mnt.box(:,idx)= [ii;9-jj];
  end
end
mnt.box(:,end+1)= [9;8];
mnt.box_sz= ones(size(mnt.box));
epo.clab= mnt.clab;

epo= proc_jumpingMeans(epo, 10);  %% brute force downsampling to 100Hz
epo= proc_baseline(epo, 250, 'beginning');
%grid_plot(epo, mnt);
epo_rsq= proc_r_square(epo);
grid_plot(epo_rsq, mnt);

spec= proc_spectrum(epo, [3 45]);
grid_plot(spec, mnt);
spec_rsq= proc_r_square(spec);
grid_plot(spec_rsq, mnt);


%% calculate channel scores (slow potetials)
fv= proc_selectIval(epo, [1000 3500]);
fv= proc_jumpingMeans(fv, 25);
for cc= 1:length(fv.clab), fprintf('%s> ', fv.clab{cc}),
  ff= proc_selectChannels(fv, cc);
  ssp1(cc)= xvalidation(ff, 'LDA', 'xTrials',[10 10]);
end
[so,si]= sort(ssp1);
nSel= max(find(so<0.4));
for cc= 1:nSel,
  fprintf('%s> %.1f%%\n', fv.clab{si(cc)}, 100*so(cc));
end
%[dmy, idx, ssp2]= proc_fs_classifierWeights(fv, 100);

ff= proc_selectChannels(fv, fv.clab(si(1:nSel)));

model_RLDA= struct('classy', 'RLDA');
model_RLDA.param= [0 0.01 0.1 0.3 0.5 0.7];


model_RDA= struct('classy', 'RDA', ...
                  'msDepth',2, 'inflvar',2);
model_RDA.param(1)= struct('index',2, ...
                           'value',[0 0.05 0.25 0.5 0.75 0.9 1]);
model_RDA.param(2)= struct('index',3, ...
                           'value',[0 0.001 0.01 0.1 0.3 0.5 0.7]);

opt_xv= struct('out_trainloss',1, 'outer_ms',1, 'xTrials',[10 10], ...
               'verbosity',2);
xvalidation(ff, model_RDA, opt_xv);
%% -> {'RDA',, 0.825, 0.2} -> 15%
%% -> {'RDA',, .75,.1} -> 15%

%% spectral features
fv= proc_selectIval(epo, [1000 3000]);
fv= proc_selectChannels(fv, 'ch_4_2','ch_5_3','ch_5_4','ch_5_5','ch_6_4','ch_6_5','ch_6_6','ch_7_4','ch_7_5','ch_8_5');
fv= proc_fourierBandMagnitude(fv, [4 40], 200);
fv= proc_jumpingMeans(fv, 2);

for ii= 1:size(fv.x,1),
  ff= fv;
  ff.x= ff.x(ii,:,:);
  srh1(ii)= xvalidation(ff, 'LDA', 'xTrials',[10 10]);
end
[so,si]= sort(srh1);
nSel= max(find(so<0.4));
for cc= 1:nSel,
  fprintf('%d> %.1f%%\n', floor(fv.t(si(cc))), 100*so(cc));
end

nSel= 5;
ff.x= fv.x(si(1:nSel),:,:);
ff.t= fv.t(si(1:nSel));
xvalidation(ff, model_RLDA, opt_xv);
%% 14%
%% 13.4%

fv= proc_selectIval(epo, [1000 3000]);
fv= proc_selectChannels(fv, 'ch_4_2','ch_5_3','ch_5_4','ch_5_5','ch_6_4','ch_6_5','ch_6_6','ch_7_4','ch_7_5','ch_8_5');
fv= proc_fourierCourseOfBandEnergy(fv, [8 12], kaiser(50,2), 25);
fv= proc_logarithm(fv);
fv= proc_meanAcrossTime(fv);
xvalidation(fv, model_RLDA, opt_xv);
%% 13%
%% 12.5%


fv= proc_selectIval(epo, [1000 3000]);
fv= proc_selectChannels(fv, 'ch_4_2','ch_5_3','ch_5_4','ch_5_5','ch_6_4','ch_6_5','ch_6_6','ch_7_4','ch_7_5','ch_8_5');
fv1= proc_fourierCourseOfBandEnergy(fv, [8 12], kaiser(50,2), 25);
fv1= proc_logarithm(fv1);
fv1= proc_meanAcrossTime(fv1);
fv2= proc_fourierCourseOfBandEnergy(fv, [15 22], kaiser(50,2), 25);
fv2= proc_logarithm(fv2);
fv2= proc_meanAcrossTime(fv2);
fv= proc_catFeatures(fv1, fv2);
xvalidation(fv, model_RLDA, opt_xv);
%% 11%
%% 10.1%

fv= proc_selectIval(epo, [1000 3000]);
fv= proc_selectChannels(fv, 'ch_4_2','ch_5_3','ch_5_4','ch_5_5','ch_6_4','ch_6_5','ch_6_6','ch_7_4','ch_7_5','ch_8_5');
fv= proc_spectrum(fv, [8 25], kaiser(50,2), 25);
xvalidation(fv, model_RLDA, opt_xv);
%% 12%


fv= proc_filtByFFT(epo, [7 13], 50);
fv= proc_selectIval(fv, [1000 3000]);
%removing those channels makes patterns nicer, but does not improve accuracy
%fv= proc_selectChannels(fv, 'not', 'ch_1_1', 'ch_2_1');
proc= ['[fv, csp_w]= proc_csp(fv, 2); ' ...
       'fv= proc_variance(fv); ' ...
       'fv= proc_logarithm(fv);'];
xvalidation(fv, 'LDA', 'proc',proc);
%% 12%


fv= proc_filtByFFT(epo, [7 13], 50);
fv.clab(:)= {'band1'};
fv2= proc_filtByFFT(epo, [15 23], 50);
fv2.clab(:)= {'band2'};
fv= proc_appendChannels(fv, fv2);
clear fv2
fv= proc_selectIval(fv, [1000 3000]);
proc= ['fv1= proc_selectChannels(fv, ''band1''); ' ...
       '[fv1, csp_w1]= proc_csp(fv1, 2); ' ...
       'fv1= proc_variance(fv1); ' ...
       'fv2= proc_selectChannels(fv, ''band1''); ' ...
       '[fv2, csp_w2]= proc_csp(fv2, 2); ' ...
       'fv2= proc_variance(fv2); ' ...
       'fv= proc_catFeatures(fv1, fv2); ' ...
       'fv= proc_logarithm(fv);'];
xvalidation(fv, 'LDA', 'proc',proc);
%% 12%


fv= proc_selectChannels(epo, 'ch_4_2','ch_5_3','ch_5_4','ch_5_5','ch_6_4','ch_6_5','ch_6_6','ch_7_4','ch_7_5','ch_8_5');
fv= proc_filtByFFT(fv, [5 35], 50);
fv= proc_selectIval(fv, [1000 3000]);
fv= proc_arCoefsPlusVar(fv, 4);
xvalidation(fv, model_RLDA, opt_xv);
%% 18%
