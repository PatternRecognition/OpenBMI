fprintf('Read the code of this script and execute it blockwise.\n');

file= 'Klaus_03_10_29/imagKlaus';
[cnt,mrk,mnt]= loadProcessedEEG(file);

%% Select just two classes for classification
mrk= mrk_selectClasses(mrk, 'left','right');

%% Define a model for classification
model_RLDA= struct('classy','RLDA', 'msDepth',2, 'inflvar',2);
model_RLDA.param= [0 0.001 0.005 0.01 0.05 0.1 0.3];

%% Common Spatial Pattern (CSP) method.
%% Note: The CSP algorithm uses label information. So you must not
%% apply CSP in advance and then do the cross validation, as this
%% procedure would bias the result. Rather the CSP has to be calculated
%% within the cross-validation on each training set.
%% See also demos/demo_validation_csp
band= [7 32];
[b,a]= butter(5, band/cnt.fs*2);
cnt_flt= proc_filt(cnt, b, a);
epo= makeEpochs(cnt_flt, mrk, [750 3500]);
fv= proc_selectChannels(epo, {'not','E*','Fp*','AF*','I*','OI*',...
                    'OPO*','TP9,10','T9,10','FT9,10'});
proc= struct('memo', 'csp_w');
proc.train= ['[fv,csp_w]= proc_csp(fv, 2); ' ...
             'fv= proc_variance(fv); ' ...
             'fv= proc_logarithm(fv);'];
proc.apply= ['fv= proc_linearDerivation(fv, csp_w); ' ...
             'fv= proc_variance(fv); ' ...
             'fv= proc_logarithm(fv);'];
xvalidation(fv, 'LDA', 'proc',proc);


%% with band power
band= [8 15];
epo= makeEpochs(cnt, mrk, 750 + [0 2000]);
epo= proc_laplace(epo);
fv= proc_selectChannels(epo, 'CFC5-6','C5-6','CCP5-6','CP5-6', 'P1-2');
fv= proc_fourierBandEnergy(fv, band, 200);
xvalidation(fv, model_RLDA);


%% with band power (calculated at 2Hz resolution)
band= [8 15];
epo= makeEpochs(cnt, mrk, [750 3000]);
epo= proc_laplace(epo);
fv= proc_selectChannels(epo, 'CFC5-6','C5-6','CCP5-6','CP5-6', 'P1-2');
fv= proc_fourierCourseOfBandEnergy(fv, band(1,:), kaiser(fv.fs/2,2), fv.fs/4);
fv= proc_logarithm(fv);
fv= proc_meanAcrossTime(fv);
xvalidation(fv, model_RLDA);


%% with band power, combined from two frequency bands (mu and beta)
band= [8 13; 17 32];
epo= makeEpochs(cnt, mrk, 750 + [0 2000]);
epo= proc_laplace(epo);
fv= proc_selectChannels(epo, 'CFC5-6','C5-6','CCP5-6','CP5-6', 'P1-2');
fv1= proc_fourierBandEnergy(fv, band(1,:), 200);
fv2= proc_fourierBandEnergy(fv, band(2,:), 200);
fv= proc_catFeatures(fv1, fv2);
xvalidation(fv, model_RLDA);

%% with power spectrum
band= [8 30];
epo= makeEpochs(cnt, mrk, 750 + [0 2000]);
epo= proc_laplace(epo);
fv= proc_selectChannels(epo, 'CFC5-6','C5-6','CCP5-6','CP5-6', 'P1-2');
fv= proc_fourierBandMagnitude(fv, band, 200);
fv= proc_jumpingMeans(fv, 4);
xvalidation(fv, model_RLDA);


%% power spectrum, 2nd version
band= [8 30];
epo= makeEpochs(cnt, mrk, [750 3000]);
epo= proc_laplace(epo);
fv= proc_selectChannels(epo, 'CFC5-6','C5-6','CCP5-6','CP5-6', 'P1-2');
fv= proc_spectrum(fv, band, kaiser(fv.fs/2,2), fv.fs/4);
xvalidation(fv, model_RLDA);


%% with AR coefficients
band= [5 35];
[b,a]= getButterFixedOrder(band, cnt.fs, 5);
cnt_flt= proc_filt(cnt, b, a);
epo= makeEpochs(cnt_flt, mrk, [750 3000]);
epo= proc_laplace(epo);
fv= proc_selectChannels(epo, 'CFC3-4','C5-6','CCP5-6','CP5-6');
fv= proc_arCoefsPlusVar(fv, 4);
xvalidation(fv, model_RLDA);
