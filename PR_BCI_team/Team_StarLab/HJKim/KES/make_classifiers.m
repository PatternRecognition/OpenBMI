function clfs =  make_classifiers(filepath, channels)

selChan = channels;%[3 5 7 12 14 16 23 25 27 28 30 32];
fs = 100;
ival = [-200 800];
baseTime =[-200 0];
selTime =[0 800];
freq = [0.5 40];
field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
marker={'2', 'non_target';'1','target'};
nFeatures = 10;

[EEG.data, EEG.marker, EEG.info]=Load_EEG(filepath,{'device','brainVision';'marker', marker;});
cnt=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
cnt=prep_selectChannels(cnt, {'Index', selChan});
smt=prep_segmentation(cnt, {'interval', ival});
smt=prep_baseline(smt, {'Time', baseTime});
smt=prep_selectTime(smt, {'Time', selTime});

%% classifiers
% 1. class ±¸ºÐ
fv = func_featureExtraction(smt, {'feature', 'erpmean'; 'nMeans', nFeatures});
[nDat, nTrials, nChans]=size(fv.x);
fv.x=reshape(permute(fv.x, [1 3 2]), [nDat*nChans nTrials]);
[clf_param]=func_train(fv,{'classifier', 'LDA'});

clfs =clf_param;

% smt.x1 = permute(smt.x, [1, 3, 2]);
% epo.x = smt.x1;
% epo.y = smt.y_logic;
% epo.clab = smt.chan;
% 
% [fv, ~, ~]=proc_multicsp(epo, 5);
% fv = proc_variance(fv);
% fv = proc_logarithm(fv);
% fv.x = reshape(fv.x, [20, 2400]);
% fv.y_dec = smt.y_dec;
% fv.y_logic = smt.y_logic;
% fv.y_class = smt.y_class;
% fv.class = smt.class;
% fv.chan = smt.chan;
% [clf_param]=func_train(fv,{'classifier', 'LDA'});
% 
% clfs =clf_param;

end