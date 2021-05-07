for j=1:10
xTrain = permute(fv_Tr{j,1}.x,[1,2,4,3]);
yTrain = categorical(fv_Tr{j,1}.event.desc);
xTest = permute(fv_Te{j,1}.x,[1,2,4,3]);
yTest = categorical(fv_Te{j,1}.event.desc);
%% define architecture
inputSize = [size(fv_Tr{j,1}.x,1), size(fv_Tr{j,1}.x,2),1];
classes = unique(fv_Tr{j,1}.event.desc);
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
    'Verbose',false, ...%'Plots','training-progress', ...
    'MaxEpochs', 1000);
net = trainNetwork(xTrain,yTrain,layers,options);

%% Test
y_pred = classify(net,xTest);
acc(j) = sum(double(y_pred == yTest)/numel(yTest));
% disp(sprintf('Test accuracy: %d %%',floor(acc*100)));
end
mean_acc = mean(acc);
disp(mean_acc)