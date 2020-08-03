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

testfile =  [EEG_IMPORT_DIR 'bci_competition_iii/tuebingen/Competition_test'];
S = load(testfile);
epo_test= struct('x', permute(S.X, [3 2 1]));
epo_test.fs= 1000;
epo_test.t= 501:3500;
epo_test.y = ones(1,size(epo_test.x,3));
epo_test.className = {'test'};
clear S
epo_test.clab= mnt.clab;


%epo_test= proc_jumpingMeans(epo_test, 10);  %% brute force downsampling to 100Hz
%epo_test= proc_baseline(epo_test, 250, 'beginning');
epo1 = proc_appendEpochs(epo,epo_test);

epo1= proc_jumpingMeans(epo1, 10);  %% brute force downsampling to 100Hz
%epo1 = proc_normalize(epo1,struct('dim',2));
epo1= proc_baseline(epo1, 250, 'beginning');
epo = proc_selectClasses(epo1,epo.className);
epo_test = proc_selectClasses(epo1,epo_test.className);

grid_plot(epo1, mnt);
epo_rsq= proc_r_square(epo1);
grid_plot(epo_rsq, mnt);


spec= proc_spectrum(epo1, [3 45]);
grid_plot(spec, mnt);
spec_rsq= proc_r_square(spec);
grid_plot(spec_rsq, mnt);



%% calculate channel scores (slow potentials)
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

model_RLDA.classy = {'probCombiner',1};
fv = proc_combineFeatures(fv1,fv2);
xvalidation(fv, model_RLDA, opt_xv);
% 9.9%

fv = proc_combineFeatures(fv1,ff);
xvalidation(fv,model_RLDA, opt_xv);
% 9.4%

fv = proc_combineFeatures(fv2,ff);
xvalidation(fv,model_RLDA, opt_xv);
% 12%

fv = proc_combineFeatures(fv2,ff);
fv = proc_combineFeatures(fv,fv1);
xvalidation(fv,model_RLDA, opt_xv);
% 9.6%

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


%%%%%% plot csp-channels against each other.
[fv,w,la] = proc_csp(epo,1);
fv = proc_variance(fv);
fv = proc_logarithm(fv);
clf;hold on;
x = squeeze(fv.x);
ind = find(fv.y(1,:));
plot(x(1,ind),x(2,ind),'xr');
%plot(x(1,ind(71:end)),x(2,ind(71:end)),'xk');
ind = find(fv.y(2,:));
plot(x(1,ind),x(2,ind),'xg');
%plot(x(1,ind(71:end)),x(2,ind(71:end)),'xc');

fv_test = proc_linearDerivation(epo_test,w);
fv_test = proc_variance(fv_test);
fv_test = proc_logarithm(fv_test);
x = squeeze(fv_test.x);
plot(x(1,:),x(2,:),'kx');
legend(fv.className{:},'testdata');


%%% plot selected channel bandpowers against each other:
fv= proc_selectIval(epo, [1000 3000]);
fv= proc_selectChannels(fv, 'ch_6_5','ch_7_5');
fv1= proc_fourierCourseOfBandEnergy(fv, [15 23], kaiser(50,2), 25);
fv1= proc_logarithm(fv1);
fv1= proc_meanAcrossTime(fv1);
x = squeeze(fv1.x);
ind = find(fv1.y(1,:));
clf;hold on;
plot(x(1,ind),x(2,ind),'xr');
ind = find(fv1.y(2,:));
plot(x(1,ind),x(2,ind),'xg');

fv_test= proc_selectIval(epo_test, [1000 3000]);
fv_test= proc_selectChannels(fv_test, 'ch_6_5','ch_7_5');
fv2= proc_fourierCourseOfBandEnergy(fv_test, [15 23], kaiser(50,2), 25);
fv2= proc_logarithm(fv2);
fv2= proc_meanAcrossTime(fv2);
x = squeeze(fv2.x);
plot(x(1,:),x(2,:),'xk');
legend(fv1.className{:},'testdata')

%% Visual inspection: 
% test data are on a completely different scale.->normalization required.
epo2 = proc_normalize(epo,struct('dim',2));
epo3 = proc_normalize(epo_test, struct('dim',2));
epo2 = proc_appendEpochs(epo2,epo3);
clear epo3;

showERPscalps(epo2,mnt,1000:200:2000);
epo = proc_selectClasses(epo2,epo.className);
epo_test = proc_selectClasses(epo2,epo_test.className);

% interesting: peak at 30 Hz which didn't occur in training.
% is this the peak which appeared at 35 Hz in training?

% try different labelling; based on slow and on oscillatory features. 
% compare the labels.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% labelling on slow features
model_RLDA.classy = 'RLDA';

%% calculate channel scores (slow potentials)
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
%[dmy, idx, ssp2]= proc_fs_classifierWeights(fv, 100);

ff= proc_selectChannels(fv, fv.clab(si(1:nSel)));
C = trainClassifier(ff,model_RLDA);

fv= proc_selectIval(epo_test, [1000 3500]);
fv= proc_jumpingMeans(fv, 25);
ff_test = proc_selectChannels(fv, fv.clab(si(1:nSel)));

out = applyClassifier(ff_test,model_RLDA,C);
%add these labels to the plots above:
plot(x(1,find(out<0)),x(2,find(out<0)),'xb');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% labelling on oscillatory features:
[fv,w,la] = proc_csp(epo,1);
fv = proc_variance(fv);
fv = proc_logarithm(fv);
C = trainClassifier(fv,model_RLDA);
fv = proc_linearDerivation(epo_test,w);
fv = proc_variance(fv);
fv = proc_logarithm(fv);
out = applyClassifier(fv, model_RLDA,C);

epo3 = proc_selectEpochs(epo_test,find(out>0));%tongue
epo3.className = {'tongue'};
epo4 = proc_selectEpochs(epo_test,find(out<0));%left
epo4.className = {'left'};
epo3 = proc_appendEpochs(epo4,epo3);
showERPscalps(epo3,mnt,1000:200:2000);
epo_rsq= proc_r_square(epo3);
grid_plot(epo_rsq, mnt);

