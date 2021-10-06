%% Input setting
% raw data
input_train  = epo_train;
input_test = epo_test;

% feature vecture 
% input_train  = fv_Tr;
% input_test = fv_Te;
%%
for j=1:15 % the number of subjects
ans_auc(:,j) = input_test{j}.y(1,:)';
xTrain = permute(input_train{j}.x,[1,2,4,3]);
yTrain = categorical(input_train{j}.event.desc);
xTest = permute(input_test{j}.x,[1,2,4,3]);
yTest = categorical(input_test{j}.event.desc);
%% define architecture
inputSize = [size(xTrain,1), size(xTrain,2),1];
classes = unique(yTrain);
numClasses = length(classes);

layers = [
    imageInputLayer(inputSize)
    convolution2dLayer(1,5, 'padding',1) % layer1
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2) 
    convolution2dLayer(3,3, 'padding',1) % layer2
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,3, 'padding',1) % layer 3
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2) 
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

%% Training
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'Verbose',false, ...
    'Plots','training-progress', ...
    'MaxEpochs', 100);
net = trainNetwork(xTrain,yTrain,layers,options);

%% Test
[y_pred, scores] = classify(net,xTest);
prop(:,j) = scores(:,2);
[~,~,~,AUC(j)] = perfcurve(ans_auc(:,j),prop(:,j),0);
acc(j) = sum(double(y_pred == yTest)/numel(yTest));
% disp(sprintf('Test accuracy: %d %%',floor(acc*100)));
end
mean_acc = mean(acc);
fprintf('Accuracy: %.4f\n',mean_acc)

mean_AUC = mean(AUC);
fprintf(mean_AUC)
