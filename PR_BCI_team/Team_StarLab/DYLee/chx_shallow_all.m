%% Converting
%% 분리
EEG_ct=EEG(:, 1:24576);
ds = downsample(EEG_ct,8);
class_all=EEG(:,24578);
% imagined pronounced 분리
n=1;
m=1;
for i=1: size(EEG,1) 
    if EEG(i, 24577)==1
        EEG_imagined(n,:)=EEG_ct(i,:);
        class_i(n,1)=class_all(i,1);
        n=n+1;
    else
        EEG_pronounced(m,:)=EEG_ct(i,:);
        class_p(m,1)=class_all(i,1);
        m=m+1;
    end
end

%% chtimeXTrials 3d
EEG_i=zeros(1,size(EEG_imagined,2),size(EEG_imagined,1));
for i=1: size(EEG_imagined,1)
    EEG_i(:,:,i)=EEG_imagined(i,:);
end

EEG_p=zeros(1,size(EEG_pronounced,2),size(EEG_pronounced,1));
for i=1: size(EEG_pronounced,1)
    EEG_p(:,:,i)=EEG_pronounced(i,:);
end




%% 4D
EEG_4i=reshape(EEG_i,1, size(EEG_i,2), 1, size(EEG_i,3));
EEG_4itrans=permute(EEG_4i,[2 1 3 4]);
ds = downsample(EEG_4itrans,8);
EEG_4ids=permute(ds,[2 1 3 4]);

%%
% XTrain=EEG_4i;
XTrain=EEG_4ids;
YTrain=class_i;

%% classication
layers = [ ...
    imageInputLayer([1 3072 1])
    convolution2dLayer([1 500],250,'Padding','same')
    convolution2dLayer([6 80],80,'Padding','same')
    batchNormalizationLayer
    eluLayer
    averagePooling2dLayer([1 75],'Stride',[15 1])
%     convolution2dLayer([40 30],30,'Padding','same')
    dropoutLayer(0.1)
    fullyConnectedLayer(11)
    softmaxLayer
    classificationLayer];

maxEpochs = 10;
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
fold=cvpartition(class_i,'kfold',kfold);
% Pre
Afold=zeros(kfold,1); confmat=0;
% Start deep learning
for i=1:kfold
  % Call index of training & testing sets
  trainIdx=fold.training(i); testIdx=fold.test(i);
  % Call training & testing features and classs
  XTrain=EEG_4ids(:,:,1,trainIdx); YTrain=class_i(trainIdx);
  XTest=EEG_4ids(:,:,1,testIdx); YTest=class_i(testIdx);
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
