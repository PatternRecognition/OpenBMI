file= 'Gabriel_00_09_05/selfpaced2sGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file);
cnt= proc_selectChannels(cnt, 'not','E*');
tcl= -120;

%% do the preprocessing
epo= makeEpochs(cnt, mrk, [-1270 0] + tcl);
fv= proc_filtBruteFFT(epo, [0.8 3], 128, 500);
fv= proc_jumpingMeans(fv, 5);

%% just for a test, try cross-validation for LDA
xvalidation(fv, 'LDA');

%% linear programming machine
model_LPM= struct('classy','LPM', 'msDepth',2, 'std_factor',2);
model_LPM.param= struct('index',2, 'scale','log', ...
                        'value', [-2:3]);
%% sparse Fisher (1)
model_FDlwlx= struct('classy','FDlwlx', 'param',[-2:3]);

%% sparse Fisher (2)
model_FDlwqx= struct('classy','FDlwqx', 'param',[-2:3]);

%% for the given model, select parameters by cross-validation
model= model_LPM;
classy= select_model(fv, model);

%% train a classifier on the whole data set
C= trainClassifier(fv, classy);

%% plot classifier as matrix
plot_classifierImage(C, fv);


%% classifier as series of scalp maps
%colormap(green_white_red(11,0.9));
%plot_patternsOfLinClassy(mnt, C, fv);
%colormap('default');
