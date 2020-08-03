file= 'bci_competition_ii/graz_train';


[epo, mrk, mnt]= loadProcessedEEG(file);

xTrials= [1 1];
msTrials= [1 1 sum(any(epo.y))-2];
warning('off', 'bbci:validation')


%% with AR coefficients
fv= proc_selectIval(epo, [3500 6000]);
fv= proc_rcCoefsPlusVar(fv, 8);
doXvalidation(fv, 'LDA', xTrials);
%% 12.1±0.0%, [train:  6.6±0.0%]  (0.7 s for [1 1] trials)

model= struct('classy','RLDA', 'msDepth',2, 'inflvar',2, 'msTrials',[1 1]);
model.param= [0 0.001 0.01 0.1];
classy= selectModel(fv, model, msTrials);
doXvalidation(fv, classy, xTrials);
%% 12.1±0.0%, [train:  6.6±0.0%]
%doXvalidationPlus(fv, model, xTrials); %% the real thing
%% 13.6±0.0%, [train:  6.7±0.0%]  (2025.6 s for [1 1] trials)


%% with power spectrum
band= [8 14];
fv= proc_selectIval(epo, [4000 5000]);
fv= proc_fourierBandMagnitude(fv, band, 128);
doXvalidation(fv, 'LDA', xTrials);
%% 14.3±0.0%, [train:  8.0±0.0%]

fv1= proc_selectIval(epo, [3500 4500]);
fv1= proc_fourierBandMagnitude(fv1, band, 128);
fv2= proc_selectIval(epo, [4500 5500]);
fv2= proc_fourierBandMagnitude(fv2, band, 128);
fv= proc_catFeatures(fv1, fv2);

model= struct('classy','RLDA', 'msDepth',3, 'inflvar',2, 'msTrials',[1 1]);
model.param= [0 0.01 0.1 0.5 0.8];
classy= selectModel(fv, model, msTrials);
doXvalidation(fv, classy, xTrials);
%% 13.6±0.0%, [train: 11.4±0.0%]
%doXvalidationPlus(fv, model, xTrials); %% the real thing


%% with mu band power
band= [8 14];
fv= proc_selectIval(epo, [4000 5000]);
fv= proc_fourierBandEnergy(fv, band, 128);
doXvalidation(fv, 'LDA', xTrials);
%% 19.3±0.0%, [train: 19.4±0.0%]


band= [7 14];  %% mu only
band= [15 35];  %% beta only
[b,a]= getButterFixedOrder(band, epo.fs, 6);
fv= proc_filt(epo, b, a);
%% do anything ...
