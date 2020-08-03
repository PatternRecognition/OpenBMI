file= 'Klaus_03_11_04/imagKlaus';

[cnt, mrk, mnt]= loadProcessedEEG(file);
cnt= proc_selectChannels(cnt, 'not','E*');
%mrk= mrk_selectClasses(mrk, 'left','right');


%% regularized LDA
model_RLDA= struct('classy', 'RLDA');
model_RLDA.param= [0 0.001 0.01 0.1 0.3 0.5 0.7];

%% linear programming machine
model_LPM= struct('classy','LPM', 'msDepth',2, 'std_factor',2);
model_LPM.param= struct('index',2, 'scale','log', ...
                        'value', [-2:3]);
%% sparse Fisher (1)
model_FDlwlx= struct('classy','FDlwlx', 'param',[-2:3]);

%% sparse Fisher (2)
model_FDlwqx= struct('classy','FDlwqx', 'param',[-2:3]);



%% do the preprocessing (power spectrum at 2Hz resolution)
band= [8 30];
epo= makeEpochs(cnt, mrk, [750 3000]);
%% spatial Laplace filter (can be omitted??)
epo= proc_laplace(epo);
%% choose only a subset of channels
fv= proc_selectChannels(epo, 'CFC5-6','C5-6','CCP5-6','CP5-6', 'P1-2');
fv= proc_spectrum(fv, band, 'win',kaiser(fv.fs/2,2), 'step',fv.fs/4);



%% just for a test, try cross-validation for RLDA
opt= struct('verbosity',2, 'outer_ms',1, 'msTrials',[3 10 -1]);
xvalidation(fv, model_RLDA, opt);


%% for the given model, select parameters by cross-validation
model= model_LPM;
classy= select_model(fv, model);

%% train a classifier on the whole data set
C= trainClassifier(fv, classy);

%% plot classifier as matrix
plot_classifierImage(C, fv);


%% classifier as series of scalp maps
colormap(green_white_red(11,0.9));
head= mnt;
head.x= 1.3*head.x;
head.y= 1.3*head.y;
plot_patternsOfLinClassy(head, C, fv);
%colormap('default');






return


%%% some alternative preprocessings


%% CSP
csp.band= [7 30];
csp.ival= [750 3000];
csp.clab= {'F7-8','FC#','FT7,8','CFC#','C#','T7,8', ...
           'CCP#','CP#','TP7,8','PCP#','P7-8'};
csp.nPat= 2;
csp.filtOrder= 5;

[b,a]= getButterFixedOrder(csp.band, cnt.fs, csp.filtOrder);
cnt_flt= proc_filt(cnt, b, a);
fv= makeEpochs(cnt_flt, mrk, csp.ival);
fv= proc_selectChannels(fv, csp.clab);
fv.proc= ['fv= proc_csp(epo, ' int2str(csp.nPat) '); ' ...
          'fv= proc_variance(fv); ' ...
          'fv= proc_logarithm(fv);'];
xvalidation(fv, 'LDA');


%% with power spectrum, 2nd version
band= [8 30];
epo= makeEpochs(cnt, mrk, 750 + [0 2000]);
epo= proc_laplace(epo);
fv= proc_selectChannels(epo, 'CFC5-6','C5-6','CCP5-6','CP5-6', 'P1-2');
fv= proc_fourierBandMagnitude(fv, band(1,:), 200);
fv= proc_jumpingMeans(fv, 4);



%% with band power (mu and beta)
band= [8 13; 17 32];
epo= makeEpochs(cnt, mrk, 750 + [0 2000]);
epo= proc_laplace(epo);
fv= proc_selectChannels(epo, 'CFC5-6','C5-6','CCP5-6','CP5-6', 'P1-2');
fv1= proc_fourierBandEnergy(fv, band(1,:), 200);
fv2= proc_fourierBandEnergy(fv, band(2,:), 200);
fv= proc_catFeatures(fv1, fv2);



%% with band power (calculated at 2Hz resolution)
band= [8 15; 17 32];
epo= makeEpochs(cnt, mrk, [750 3000]);
epo= proc_laplace(epo);
fv= proc_selectChannels(epo, 'CFC5-6','C5-6','CCP5-6','CP5-6', 'P1-2');
fv1= proc_fourierCourseOfBandEnergy(fv, band(1,:), kaiser(fv.fs/2,2), fv.fs/4);
fv1= proc_logarithm(fv1);
fv1= proc_meanAcrossTime(fv1);
fv2= proc_fourierCourseOfBandEnergy(fv, band(2,:), kaiser(fv.fs/2,2), fv.fs/4);
fv2= proc_logarithm(fv2);
fv2= proc_meanAcrossTime(fv2);
fv= proc_catFeatures(fv1, fv2);
