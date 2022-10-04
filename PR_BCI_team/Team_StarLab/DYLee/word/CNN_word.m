%% 데이터 입력형태 바꾸기
data_processing;
%% 

EEG_3itrans=permute(EEG_3i,[2 1 3]);
ds = downsample(EEG_3itrans,8);
EEG_3ids=permute(ds,[2 1 3 ]);


n=1;
% for i=1: size(EEG_3ids,3)
for i=1: size(EEG_3i,3)
    if class_i(i)>5
%          class_word(n)=class_i(i);
        class_word(n)=class_i(i)-5;
%         EEG_3word(:, :, n)=EEG_3ids(:, :, i);
        EEG_3word(:, :, n)=EEG_3i(:, :, i);
        n=n+1;
    end
end


%% 4D
EEG_4i=reshape(EEG_3word,6, size(EEG_3word,2), 1, size(EEG_3word,3));
% EEG_4i=reshape(EEG_3word,6, 4096, 1, 304);
EEG_4itrans=permute(EEG_4i,[2 1 3 4]);
ds = downsample(EEG_4itrans,8);
EEG_4ids=permute(ds,[2 1 3 4]);
% rds=ds';
% ds=proc_downsample(EEG_4i, 128);
%%
XTrain=EEG_4i;
% XTrain=EEG_4ids;
YTrain=class_word;

%% classication
layers = [ ...
    imageInputLayer([6 512 1])
%       imageInputLayer([6 4096 1])
%     imageInputLayer([64 2100 1])
%     imageInputLayer([80 1280 1])
%     convolution2dLayer(5,20)
    convolution2dLayer([1 5],20,'Padding','same')
    leakyReluLayer
      convolution2dLayer([6 1],20,'Padding','same')
    leakyReluLayer
      convolution2dLayer([1 5],20,'Padding','same')
    leakyReluLayer
      convolution2dLayer([1 3],40,'Padding','same')
    leakyReluLayer
      convolution2dLayer([1 3],100,'Padding','same')
    leakyReluLayer
      convolution2dLayer([1 3],250,'Padding','same')
    leakyReluLayer
    convolution2dLayer([1 3],500,'Padding','same')
    leakyReluLayer
%     maxPooling2dLayer(2,'Stride',2)
    dropoutLayer(0.1)
    fullyConnectedLayer(6)
    softmaxLayer
    classificationLayer];

maxEpochs = 50;
miniBatchSize =32;

% maxEpochs = 50;
% miniBatchSize = 50;

options = trainingOptions('adam', ...
    'InitialLearnRate',0.001,...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',false, ...
    'Plots','training-progress', 'ExecutionEnvironment','gpu' );


%%
% Start cross validation 
rng('default'); 
% Divide data into k-folds
% fold=cvpartition(label,'kfold',kfold);
kfold=5;
fold=cvpartition(class_word,'kfold',kfold);
% Pre
Afold=zeros(kfold,1); confmat=0;
% Start deep learning
for i=1:kfold
  % Call index of training & testing sets
  trainIdx=fold.training(i); testIdx=fold.test(i);
  % Call training & testing features and classs
  XTrain=EEG_4i(:,:,1,trainIdx); YTrain=class_word(trainIdx);
  XTest=EEG_4i(:,:,1,testIdx); YTest=class_word(testIdx);
  % Convert class of both training and testing into categorial format
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
fprintf('\n Classification Accuracy (CNN): %g %% \n ',Acc);

%%
YPred = predict(net,XTest);

% rmse = sqrt(mean((YTest - YPred).^2));
