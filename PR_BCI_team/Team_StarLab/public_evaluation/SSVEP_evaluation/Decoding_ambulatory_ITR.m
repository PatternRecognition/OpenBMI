total_conf=zeros(length(unique(fv{1,1}.y)));
for j=1:sum(~cellfun('isempty', cap_epo(:,1)))
%%
xTrain = fv{j,1}.x;
yTrain = categorical(fv{j,1}.y);
xTest = fv{j,2}.x;
yTest = categorical(fv{j,2}.y);
%%
% xTrain = cat(4,fv{j,1}.x, fv{j,2}.x(:,:,:,1:40));
% yTrain = cat(2,categorical(fv{j,1}.y),categorical(fv{j,2}.y(1:40)));
% xTest = fv{j,2}.x(:,:,:,41:end);
% yTest = categorical(fv{j,2}.y(41:end));
%% define architecture
inputSize = [size(xTrain,1), size(fv{j,1}.x,2),size(fv{j,1}.x,3)];
classes = unique(yTrain);
numClasses = length(classes);

layers = [
    imageInputLayer(inputSize)

    convolution2dLayer([1,size(fv{j,1}.x,2)], 4, 'padding','same') % layer3
    reluLayer
%     dropoutLayer(0.1)

    % frequency: 5.45, 8.75, 12
    convolution2dLayer([3,1], 8, 'padding','same') % layer2 [9, 1]
    reluLayer
%     dropoutLayer(0.1)

    % frequency: 5.45, 8.75, 12
    convolution2dLayer([3,1], 16, 'padding','same') % layer2 [9, 1]
    reluLayer
%     dropoutLayer(0.1)

    fullyConnectedLayer(32) % layer4
    reluLayer
    dropoutLayer(0.1)
    
    fullyConnectedLayer(numClasses)
    
    softmaxLayer
    classificationLayer];

%% Training
options = trainingOptions('rmsprop', ...
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

