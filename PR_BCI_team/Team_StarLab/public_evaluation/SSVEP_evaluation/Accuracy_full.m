clear all;
startup_bbci_toolbox

%% Data Load
% dire = 'C:\Users\young\OneDrive\2-PR?ž©\?”„ë¡œì ?Š¸\ear-EEG\analysis_code\data\2020_public_test\public_test_data';
dire = 'E:\ear_data\2020_paper_data\data_1906\SSVEP\data_publish\public_test_data';

for sub=1:10
    filename = sprintf('s%d',sub);
    epo_train{sub} = load(fullfile(dire,'train',filename)).epo;
    epo_test{sub} = load(fullfile(dire,'test',filename)).epo;
end

%% Feature extraction
% Make FFT features
for sub = 1:10
epo = epo_train{sub};
[fv_tr{sub}.x, fv_tr{sub}.y] = feature_extraction_FFT(epo);

epo = epo_test{sub};
[fv_te{sub}.x, fv_te{sub}.y] = feature_extraction_FFT(epo);

end

%% classifier
total_conf=zeros(3);

for sub=1:10
%% Training 
xTrain = fv_tr{sub}.x;
yTrain = categorical(fv_tr{sub}.y);
xTest = fv_te{sub}.x;
yTest = categorical(fv_te{sub}.y);

% define architecture
inputSize = [size(xTrain,1), size(xTrain,2),size(xTrain,3)];
classes = unique(yTrain);
numClasses = length(classes);

% build model
layers = build_model(inputSize, numClasses);

%% Training
options = trainingOptions('adam', ... %rmsprop
    'InitialLearnRate',0.001, ...
    'Verbose',false, ...
    'Plots','training-progress', ...
    'ValidationData',{xTest,yTest}, ...
    'MaxEpochs', 500);
% 'ValidationData',{xTest,yTest}, ...
% 'Plots','training-progress', ...
net = trainNetwork(xTrain,yTrain,layers,options);

%% Test 
y_pred = classify(net,xTest);
acc(sub) = sum(double(y_pred' == yTest)/numel(yTest));
fprintf('Test accuracy s%d: %d %% \n',sub,floor(acc(sub)*100));
conf{sub} = confusionmat(yTest,y_pred');
total_conf = total_conf+conf{sub};
end
mean(acc)



%% Functions

function [epo_X, epo_Y] = feature_extraction_FFT(epo)

chan = {'PO3','POz','PO4','O1','Oz','O2'};

epo = proc_selectChannels(epo, chan);
epo = proc_selectIval(epo, [0 4000]);  % [0 4000]
dataset = permute(epo.x, [3,1,2]);

[tr, dp, ch] = size(dataset); % tr: trial, dp: time, ch: channel

nominal = [];
for i=1:size(epo.y,2)
    nominal(i) = find(epo.y(:,i),1)-1;
end


%% Fast Fourier Transform (FFT)
X_arr=[]; % make empty array
for k=1:tr % trials
    x=squeeze(dataset(k, :,:)); % data
    N=length(x);    % get the number of points
    kv=0:N-1;        % create a vector from 0 to N-1
    T=N/epo.fs;         % get the frequency interval
    freq=kv/T;       % create the frequency range
    X=fft(x)/N*2;   % normalize the data
    cutOff = ceil(N/2); % get the only positive frequency
    
    % take only the first half of the spectrum
    X=abs(X(1:cutOff,:)); % absolute values to cut off
    freq = freq(1:cutOff); % frequency to cut off
    XX = permute(X,[3 1 2]);
    X_arr=[X_arr; XX]; % save in array
end

%% frequency band
% f_gt = [11 7 5];  % 5.45, 8.75, 12
f_last = find( freq > 50, 1); % freq < 30Hz
X_arr = X_arr(:,1:f_last,:); %

%% get features
epo_X = permute(X_arr,[2,3,4,1]);
epo_Y = nominal;

end



function layers = build_model(inputSize, outputSize)
layers = [
    imageInputLayer(inputSize)

    convolution2dLayer([1,inputSize(2)], 32, 'padding','same') % layer3
    reluLayer
%     dropoutLayer(0.1)

    % frequency: 5.45, 8.75, 12
    convolution2dLayer([5,1], 64, 'padding','same') % layer2 [9, 1]
    reluLayer
%     dropoutLayer(0.1)

    fullyConnectedLayer(64) % layer4
    reluLayer
    dropoutLayer(0.1)
    
    fullyConnectedLayer(outputSize)
    
    softmaxLayer
    classificationLayer];
end