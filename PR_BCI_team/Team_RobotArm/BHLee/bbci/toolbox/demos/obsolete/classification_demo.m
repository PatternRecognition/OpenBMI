file= 'Gabriel_00_09_05/selfpaced2sGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file);
epo= makeSegments(cnt, mrk, [-1300 0]-120);

nTrials= [10 10];               %% for 10 times 10-fold cross-validation
msTrials= [3 10 round(9/10*size(epo.y,2))];       %% for model selection

%% do some preprocessing
fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_filtBruteFFT(fv, [0.8 3], 128, 150);
fv= proc_jumpingMeans(fv, 5);

%% cross-validation for preprocessed data with Fisher Discriminant
doXvalidation(fv, 'FisherDiscriminant', nTrials);

%% select model parameter of classifier before cross-validation
model= struct('classy', {{'RDA', 1}}),
model.param= struct('index',3, 'value',[0 .25 .5 .75 1]);
classy= selectModel(fv, model, msTrials);
doXvalidation(fv, classy, nTrials);

%% select best preprocessing for classification with selected classifier
p_tape= 'p_nips01_scp_many';
[pn, c, errMean]= selectProcessFromTape(epo, p_tape, classy, msTrials, 1);
plot(errMean); legend('test', 'train'); drawnow;
eval(getBlockFromTape(p_tape, pn)); %% perform selected preprocessing
doXvalidation(fv, classy, nTrials);

%% select best classification model for preprocessed data
[classy, errMean]= selectModelFromTape(fv, 'c_nips01_lin', msTrials, 1);
plot(errMean); legend('test', 'train'); drawnow;
doXvalidation(fv, classy, nTrials);

%% select best preprocessing for optimal chosen classification model
[pn, c, errMean]= selectProcessFromTape(epo, ...
                    p_tape, 'c_nips01_lin', msTrials, 1);
plot(errMean); legend('test', 'train'); drawnow;
doXvalidation(fv, classy, nTrials);
