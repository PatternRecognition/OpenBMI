%% PSD-SVM
% Zhang, X. et al.
% Design of a fatigue detection system for high speed trains based on
% driver vigilance using a wireless wearable EEG.
% Sensors, 2017, 17, pp. 1--21.

clear;
close all;
clc

classNum = 4;       % 분류 하고자 하는 class의 수 (2 or 5)
subNum = 5;        % 분석 진행한 피험자 수
testDataIdx = 1;    % 4-fold 중 test set으로 사용할 데이터 번호
saveFn = [];

% 성능을 저장 할 경우 아래 코드 주석 해제 (덮어쓰기 주의!!)
% saveFn = sprintf('BrainSciData/Results/Accuracy/%d_fold_%dclass_PSD-SVM.mat', testDataIdx, classNum);

%% EEG data load

% 4-fold Data load
loadIdx = 1;
for i = 1 : 4
    if testDataIdx == i
        loadFnEEG = sprintf('BrainSciCode/Data/%d_fold_%dclass_EEG.mat', i, classNum);
        loadEEG = load(loadFnEEG);
        XXTest = loadEEG.eegTeData;
        loadFnLabel = sprintf('BrainSciCode/Data/%d_fold_%dclass_Label.mat', i, classNum);
        loadLabel = load(loadFnLabel);
        YTest = loadLabel.lableTeData;
    else
        if loadIdx == 1
            loadFnEEG = sprintf('BrainSciCode/Data/%d_fold_%dclass_EEG.mat', i, classNum);
            loadEEG = load(loadFnEEG);
            XXTrain = loadEEG.eegTeData;
            loadFnLabel = sprintf('BrainSciCode/Data/%d_fold_%dclass_Label.mat', i, classNum);
            loadLabel = load(loadFnLabel);
            YTrain = loadLabel.lableTeData;
        else
            loadFnEEG = sprintf('BrainSciCode/Data/%d_fold_%dclass_EEG.mat', i, classNum);
            loadEEG = load(loadFnEEG);
            XXTrain = [XXTrain; loadEEG.eegTeData];
            loadFnLabel = sprintf('BrainSciCode/Data/%d_fold_%dclass_Label.mat', i, classNum);
            loadLabel = load(loadFnLabel);
            YTrain = [YTrain; loadLabel.lableTeData];
        end
        loadIdx = loadIdx + 1;
    end
    clear loadFnEEG loadFnLabel loadEEG loadLabel
end
clear loadIdx i

% Data resizing for PSD feature extraction
for i = 1 : length(XXTrain)
    inputData.x(:, :, i) = XXTrain{i}';
    inputData.fs = 100;
end
for i = 1 : length(XXTest)
    inputData2.x(:, :, i) = XXTest{i}';
    inputData2.fs = 100;
end
clear XXTrain XXTest

YYYTrain = categorical(cell2mat(YTrain));
YYYTest = categorical(cell2mat(YTest));
clear YTrain YTest i j

% PSD feature extraction for training
[cca.theta, avg.theta] = PSD_2(inputData.x, [4 8], inputData.fs);
[cca.alpha, avg.alpha] = PSD_2(inputData.x, [8 13], inputData.fs);
[cca.beta, avg.beta] = PSD_2(inputData.x, [13 30], inputData.fs);
allPsdTrain = [avg.theta', avg.alpha', avg.beta'];

[cca.theta, avg.theta] = PSD_2(inputData2.x, [4 8], inputData2.fs);
[cca.alpha, avg.alpha] = PSD_2(inputData2.x, [8 13], inputData2.fs);
[cca.beta, avg.beta] = PSD_2(inputData2.x, [13 30], inputData2.fs);
allPsdTest = [avg.theta', avg.alpha', avg.beta'];

xTrainIn = allPsdTrain;
yTrainIn = double(YYYTrain);
xTestIn = allPsdTest;
yTestIn = double(YYYTest);
clear XXXTrain XXXTest YYYTrain YYYTest

%% Machin Learning (PSD-SVM)

% Model Train
svr = fitrkernel(xTrainIn,yTrainIn, 'KernelScale', 'auto', 'Learner', 'svm', 'Regularization', 'ridge');

% Model Test
predTestSvr = predict(svr, xTestIn);

% Accuracy calculation
classTestSvr = round(predTestSvr);
if classNum == 2
    misIdx = find(classTestSvr > 2);
    for i = length(misIdx)
        classTestSvr(misIdx, 1) = 2;
    end
end
acc = find(classTestSvr == yTestIn);
disp((length(acc) / length(yTestIn)));

% Performance save
if ~isempty(saveFn)
    % save(Save file location, Test label, Predicted label, Network, Corrected Data Index)
    save(saveFn, 'yTestIn', 'classTestSvr', 'net', 'acc');
end

disp("EEG 끝");

