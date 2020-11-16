%% 4D
eeg_4d=reshape(epo_all.x,64, 513, 1, 1144);

%%
XTrain=eeg_4d;
% XTrain=epo_all.x;
YTrain=y';

%%
layer = lstmLayer(100,'Name','lstm1')

numFeatures = [64 513 1];
numHiddenUnits = 100;
numClasses = 13;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

maxEpochs = 100;
miniBatchSize = 27;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(XTrain,YTrain,layers,options);

figure
plot(XTrain{1}')
title("Training Observation 1")
numFeatures = size(XTrain{1},1);
legend("Feature " + string(1:numFeatures),'Location','northeastoutside')

%%
% Start cross validation 
rng('default'); 
% Divide data into k-folds
% fold=cvpartition(label,'kfold',kfold);
kfold=10;
fold=cvpartition(y,'kfold',kfold);
% Pre
Afold=zeros(kfold,1); confmat=0;
% Start deep learning
for i=1:kfold
  % Call index of training & testing sets
  trainIdx=fold.training(i); testIdx=fold.test(i);
  % Call training & testing features and ys
  XTrain=eeg_4d(:,:,1,trainIdx); YTrain=y(trainIdx);
  XTest=eeg_4d(:,:,1,testIdx); YTest=y(testIdx);
  % Convert y of both training and testing into categorial format
  YTrain=categorical(YTrain); YTest=categorical(YTest);
%   YTrain=YTrain'; YTest=YTest';
  % Training model
  net = trainNetwork(XTrain,YTrain,layers,options);
  % Perform testing
  Pred=classify(net,XTest);
  % Confusion matrix
  con=confusionmat(YTest,Pred);
  % Store temporary
  confmat=confmat+con; 
  % Accuracy of each k-fold
  Afold(i,1)=100*sum(diag(con))/sum(con(:));
end
% Average accuracy over k-folds 
Acc=mean(Afold); 
% Store result
CNN.fold=Afold; CNN.acc=Acc; CNN.con=confmat; 
fprintf('\n classification Accuracy (CNN): %g %% \n ',Acc);

%%
YPred = predict(net,XTest);

% rmse = sqrt(mean((YTest - YPred).^2));
