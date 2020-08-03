expno= 1;

expbase= readDatabase;
idx= getExperimentIndex(expbase, 'selfpaced', '2s', expno);

[cnt, mrk, mnt]= loadProcessedEEG(expbase(idx).file);
epo= makeSegments(cnt, mrk, [-150 0]-120);

fv= proc_selectChannels(epo, 'FC#','C#','CP#');  %% channels over motor cortex
%fv= proc_selectChannels(epo, 'not', 'E*');      %% exclude EMG, EOG channels

%classy= {'LPM', '*log'};           %% linear programming machine
%classy= {'LPM', 1};                %% preselected model parameter
%classy= {'FDlwlx', '*log'};        %% linear sparse Fisher Discriminant
classy= {'FDlwlx', 0.01};
%classy= {'FDlwqx', '*log'};        %% sparse Fisher Discriminant
%classy= {'FDlwqx', 100};

%% choose regularization by model selection (if not preselected):
if getModelParameterIndex(classy)<=length(classy),
  model.classy= classy;
  model.param= -2:4;
  model.msDepth= 2;
  classy= selectModel(fv, model, [5 10], 1);
end

%% train a classifier on the whole data set
C= trainClassifier(fv, classy);

%% display the projection vector C.w as (abs) weighting of input features
displayProjectionPattern(fv, C.w);

title(['feature selection  by ' classy{1} ' for ' untex(epo.title)]);
