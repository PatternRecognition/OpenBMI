% Regression - LSTM
% written by D.-Y. Lee

clc; clear all; close all;

load('G:\2_SMC_DataConstruction_tr_MI\train\1');

mu = mean(trainX);
sig = std(trainX);
for i = 1:numel(trainX)
    trainX = (trainX - mu) ./ sig;
end

thr = 150;
for i =1:numel(trainY)
    trainY(trainY > thr) = thr;
end

for i=1:numel(trainX)
    sequence = trainX;
    sequenceLengths = size(sequence,2);
end

[sequenceLengths,idx] = sort(sequenceLengths,'descend');
trainX = trainX(idx);
trainY = trainY(idx);

figure
bar(sequenceLengths)
xlabel("Sequence")
ylabel("Length")
title("Sorted Data")

miniBatchSize = 20;


numResponses = size(trainY,1);
featureDimension = size(trainX,1);
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(featureDimension)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(50)
    dropoutLayer(0.5)
    fullyConnectedLayer(numResponses)
    regressionLayer];

maxEpochs = 60;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','training-progress',...
    'Verbose',0);

net = trainNetwork(trainX,trainY,layers,options);

load('G:\2_SMC_DataConstruction_tr_MI\test\1');

YPred = predict(net,testX,'MiniBatchSize',1);

idx = randperm(numel(YPred),4);
figure()
for i = 1:numel(idx)
    subplot(2,2,i)
    
    plot(testY{idx(i)},'--')
    hold on
    plot(YPred{idx(i)},'.-')
    hold off
    
    ylim([0 thr + 25])
    title("Test Observation " + idx(i))
    xlabel("Time Step")
    ylabel("RUL")
end
legend(["Test Data" "Predicted"],'Location','southeast')


for i = 1:numel(testY)
    testYLast = testY(end);
    YPredLast = YPred(end);
end
figure()
rmse = sqrt(mean((YPredLast - testYLast).^2))

histogram(YPredLast - testYLast)
title("RMSE = " + rmse)
ylabel("Frequency")
xlabel("Error")






