total_conf=zeros(3);
for j=1:sum(~cellfun('isempty', cap_epo(:,1)))
xTrain = fv{j,1}.x;
yTrain = categorical(fv{j,1}.y);
xTest = fv{j,2}.x;
yTest = categorical(fv{j,2}.y);
%% define architecture
inputSize = [size(xTrain,1), size(fv{j,1}.x,2),size(fv{j,1}.x,3)];
classes = unique(yTrain);
numClasses = length(classes);

layers = [
    imageInputLayer(inputSize)

    convolution2dLayer([1,size(fv{j,1}.x,2)], 32, 'padding','same') % layer3
    reluLayer
%     dropoutLayer(0.1)

    % frequency: 5.45, 8.75, 12
    convolution2dLayer([9,1], 64, 'padding','same') % layer2 [9, 1]
    reluLayer
%     dropoutLayer(0.1)

    fullyConnectedLayer(64) % layer4
    reluLayer
    dropoutLayer(0.1)
    
    fullyConnectedLayer(numClasses)
    
    softmaxLayer
    classificationLayer];

%% Training
options = trainingOptions('rmsprop', ... %rmsprop
    'InitialLearnRate',0.001, ...
    'Verbose',false, ...
    'MaxEpochs', 50);
% 'ValidationData',{xTest,yTest}, ...
% 'Plots','training-progress', ...
net = trainNetwork(xTrain,yTrain,layers,options);

%% Test 
y_pred = classify(net,xTest);
acc(j) = sum(double(y_pred' == yTest)/numel(yTest));
fprintf('Test accuracy: %d %% \n',floor(acc(j)*100));
conf{j} = confusionmat(yTest,y_pred');
total_conf = total_conf+conf{j};
end
mean(acc)

