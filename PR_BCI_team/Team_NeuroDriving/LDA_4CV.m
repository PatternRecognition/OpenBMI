function accuracy_eeg_sleep = LDA_4CV(myfeature, highIdx, lowIdx, s)


%% Classification
target = myfeature(highIdx, :);
nonTarget = myfeature(lowIdx, :);

% Creation of cross-validation set
clear trainTarget trainNon testTarget testNon;
trainTarget(1, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 4];
trainNon(1, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(1, :) = [1 : floor(size(target, 1) / 4)];
testNon(1, :) = [1 : floor(size(nonTarget, 1) / 4)];

trainTarget(2, :) = [1 : floor(size(target, 1) / 4) floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(2, :) = [1 : floor(size(nonTarget, 1) / 4) floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(2, :) = [floor(size(target, 1) / 4) + 1 : floor(size(target, 1) / 4) * 2];
testNon(2, :) = [floor(size(nonTarget, 1) / 4) + 1 : floor(size(nonTarget, 1) / 4) * 2];

trainTarget(3, :) = [1 : floor(size(target, 1) / 4) * 2 floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
trainNon(3, :) = [1 : floor(size(nonTarget, 1) / 4) * 2 floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];
testTarget(3, :) = [floor(size(target, 1) / 4) * 2 + 1 : floor(size(target, 1) / 4) * 3];
testNon(3, :) = [floor(size(nonTarget, 1) / 4) * 2 + 1 : floor(size(nonTarget, 1) / 4) * 3];

trainTarget(4, :) = [1 : floor(size(target, 1) / 4) * 3];
trainNon(4, :) = [1 : floor(size(nonTarget, 1) / 4) * 3];
testTarget(4, :) = [floor(size(target, 1) / 4) * 3 + 1 : floor(size(target, 1) / 4) * 4];
testNon(4, :) = [floor(size(nonTarget, 1) / 4) * 3 + 1 : floor(size(nonTarget, 1) / 4) * 4];

trainLabel = [zeros(1, size(trainTarget, 2)) ones(1, size(trainNon, 2));
    ones(1, size(trainTarget, 2)) zeros(1, size(trainNon, 2))];
testLabel = [ones(1, size(testTarget, 2)) zeros(1, size(testNon, 2))];



% 4-fold cross validation for EEG
for c = 1 : 4
    clear fv_eeg_train fv_eeg_test C_eeg;
    fv_eeg_train.x = [target(trainTarget(c, :), :); nonTarget(trainNon(c, :), :)]';
    fv_eeg_train.y = trainLabel;
    C_eeg = trainClassifier(fv_eeg_train, {'RLDAshrink','prior', nan, 'store_prior', 1, 'store_means', 1, ...
        'store_cov', 1, 'store_invcov', 1, 'scaling', 1});
    
    fv_eeg_test.x = [target(testTarget(c, :), :); nonTarget(testNon(c, :), :)]';
    fv_eeg_test.y = testLabel;
    pred = applyClassifier(fv_eeg_test, 'RLDAshrink', C_eeg);
    
    pTarget = find(pred > 0);
    pNon = find(pred <= 0);
    pred(pTarget) = 1; pred(pNon) = 0;
    correct = find(pred == testLabel);
    acc(c) = length(correct) / length(pred) * 100;
end
accuracy_eeg_sleep = mean(acc)