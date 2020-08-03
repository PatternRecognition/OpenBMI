% script for loading and classifying bci-competition-iii-dataset 1.
%
% EEG_IMPORT_DIR should be set to /home/neuro/data/BCI/eegImport/

% load the training data
file= [EEG_IMPORT_DIR 'bci_competition_iii/dataset_i_train'];

[epo,dum,mnt] = loadProcessedEEG(file,'',{'cnt','mnt'});

% load the test data
testfile =  [EEG_IMPORT_DIR 'bci_competition_iii/dataset_i_test'];

[epo_test,dum,mnt] = loadProcessedEEG(testfile,'',{'cnt','mnt'});

epo1 = proc_appendEpochs(epo,epo_test);

epo1= proc_jumpingMeans(epo1, 10);  %% brute force downsampling to 100Hz
epo1= proc_baseline(epo1, 250, 'beginning');
epo = proc_selectClasses(epo1,epo.className);
epo_test = proc_selectClasses(epo1,epo_test.className);

%% calculate channel scores (slow potentials);
% then: classification on slow potentials.
fv= proc_selectIval(epo, [1000 3500]);
fv= proc_jumpingMeans(fv, 25);
for cc= 1:length(fv.clab), fprintf('%s> ', fv.clab{cc}),
  ff= proc_selectChannels(fv, cc);
  ssp1(cc)= xvalidation(ff, 'LDA', 'xTrials',[10 10]);
end
[so,si]= sort(ssp1);
nSel= 9%max(find(so<0.4));
for cc= 1:nSel,
  fprintf('%s> %.1f%%\n', fv.clab{si(cc)}, 100*so(cc));
end
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


% alpha and beta band classification on selected channels(concat. features)
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

% feature combination on the same features
model_RLDA.classy = {'probCombiner',1};
fv = proc_combineFeatures(fv1,fv2);
xvalidation(fv, model_RLDA, opt_xv);
% 9.9%

% feature combination: slow potentials with alpha
fv = proc_combineFeatures(fv1,ff);
xvalidation(fv,model_RLDA, opt_xv);
% 9.4%

% feature combination: slow potentials with beta
fv = proc_combineFeatures(fv2,ff);
xvalidation(fv,model_RLDA, opt_xv);
% 12%

% feature combination: slow potentials, alpha and beta
fv = proc_combineFeatures(fv2,ff);
fv = proc_combineFeatures(fv,fv1);
xvalidation(fv,model_RLDA, opt_xv);
% 9.6%


% feature concatenation: alpha and beta, CSP patterns.
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


% feature combination: alpha and beta, CSP patterns.
proc= ['fv1= proc_selectChannels(fv, ''band1''); ' ...
       '[fv1, csp_w1]= proc_csp(fv1, 2); ' ...
       'fv1= proc_variance(fv1); ' ...
       'fv2= proc_selectChannels(fv, ''band1''); ' ...
       '[fv2, csp_w2]= proc_csp(fv2, 2); ' ...
       'fv2= proc_variance(fv2); ' ...
       'fv= proc_combineFeatures(fv1, fv2); ' ...
       'fv= proc_logarithm(fv);'];
opt_xv.proc = proc;
xvalidation(fv, model_RLDA, opt_xv);
%% 10.4%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Visual inspection: 
% test data are on a completely different scale.->normalization required.
epo2 = epo;
epo3 = epo_test;
x = reshape(permute(epo.x,[1 3 2]),[size(epo.x,1)*size(epo.x,3) size(epo.x,2)]);
scale = std(x);
x = reshape(permute(epo_test.x,[1 3 2]),[size(epo_test.x,1)*size(epo_test.x,3) size(epo_test.x,2)]);
scale_test = std(x);
x = x.*repmat(scale./scale_test,[size(epo_test.x,1)*size(epo_test.x,3) 1]);
epo3.x=permute(reshape(x',[size(epo_test.x,2) size(epo_test.x,1) size(epo_test.x,3)]),[2 1 3]);
epo2 = proc_appendEpochs(epo2,epo3);
epo = proc_selectClasses(epo2,epo.className);
epo_test = proc_selectClasses(epo2,epo_test.className);

showERPscalps(epo2,mnt,500:500:3500);
saveFigure([fig_dir 'ERP_scalps_scaled'], [10 6]*2);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% another approach: UNSUPERVISED.
fv= proc_selectIval(epo_test, [1000 3000]);
fv= proc_selectChannels(fv, 'ch_4_2','ch_5_3','ch_5_4','ch_5_5','ch_6_4','ch_6_5','ch_6_6','ch_7_4','ch_7_5','ch_8_5');
fv1= proc_fourierCourseOfBandEnergy(fv, [15 23], kaiser(50,2), 25);
fv1= proc_logarithm(fv1);
fv= proc_meanAcrossTime(fv1);

out1 = kmeans(squeeze(fv.x),2,1);
% on training data: 20.5% error.
