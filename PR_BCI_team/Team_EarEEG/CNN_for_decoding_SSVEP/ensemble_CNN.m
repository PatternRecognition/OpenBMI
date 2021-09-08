%%
startup_bbci_toolbox

% raw Î°? ?ï†Í±∞Î©¥ 
input_train  = epo_train;
input_test = epo_test;

% feature vecture Î°? ?ï†Í±∞Î©¥
% input_train  = fv_Tr;
% input_test = fv_Te;
plots = "training-progress";
%%
for j=1:15
    %%
ans_auc(:,j) = input_test{j}.y(1,:)';
nTrial = size(input_test{j}.y,2);
n_enb = 4; %size(input_train{j}.y,2)/nTrial;
nTr_enb = 48;

%% Train / Test define
% ensamble separate
[epo_t]= proc_selectClasses(input_train{j}, {'target'});
[epo_nt]= proc_selectClasses(input_train{j}, {'non-target'});


for i_enb = 1:n_enb
   if i_enb == n_enb % rest all at the last one
       tr_tn{i_enb} = proc_selectEpochs(epo_nt, (i_enb-1)*nTr_enb+1 : length(epo_nt.y));
   else
       tr_tn{i_enb} = proc_selectEpochs(epo_nt, (i_enb-1)*nTr_enb+1 : i_enb *nTr_enb);
   end
   
   tr_enb{i_enb} = proc_appendEpochs(tr_tn{i_enb}, epo_t);
   
   xTrain{i_enb} = permute(tr_enb{i_enb}.x,[1,2,4,3]);
   yTrain{i_enb} = tr_enb{i_enb}.y;
end

xTest = permute(input_test{j}.x,[1,2,4,3]);
yTest = input_test{j}.y; %event.desc

%% define architecture
inputSize = [size(xTest,1), size(xTest,2),1];
classes = unique(yTest);
numClasses = length(classes);

layers = [
    imageInputLayer(inputSize,'Normalization','none','Name','input')
    convolution2dLayer(1,5, 'padding',1,'Name','conv1') % layer1
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    maxPooling2dLayer(2,'Stride',2,'Name','mp1') 
    convolution2dLayer(3,3, 'padding',1,'Name','conv2') % layer2
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    convolution2dLayer(3,3, 'padding',1,'Name','conv3') % layer 3
    batchNormalizationLayer('Name','bn3')
    reluLayer('Name','relu3')
    maxPooling2dLayer(2,'Stride',2,'Name','mp2') 
    fullyConnectedLayer(numClasses,'Name','fc')
    softmaxLayer('Name','softmax')];

lgraph = layerGraph(layers);

%% Training
dlnet = dlnetwork(lgraph);
numEpochs = 200;
initialLearnRate = 0.01;
decay = 0.01;
momentum = 0.9;

if plots == "training-progress"
    figure
    lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
    ylim([0 inf])
    xlabel("Iteration")
    ylabel("Loss")
    grid on
end

averageGrad = [];
averageSqGrad = [];

start = tic;

% train start
for epoch = 1:numEpochs

    for i_enb = 1:n_enb
        dlX = dlarray(single(xTrain{i_enb}),'SSCB');
        % If training on a GPU, then convert data to a gpuArray.
        dlX = gpuArray(dlX);
        Y =single(yTrain{i_enb});
        [grad{i_enb},loss{i_enb}] = dlfeval(@modelGradients,dlnet,dlX,Y);
    end
%         loss = crossentropy(dlX,Y);
    % n_enb = 4
    grad_enb = grad{1};
    for k = 1:size(grad_enb,1)
        grad_enb.Value{k} = (grad{1}.Value{k} + grad{2}.Value{k} + grad{3}.Value{k} + grad{4}.Value{k})/4; % ?ù¥Í±∏Î°ú ensemble ?ïòÍ∏?
    end
    loss_enb = (loss{1} + loss{2} + loss{3} + loss{4})/4; 
    
    [dlnet,averageGrad,averageSqGrad] = adamupdate(dlnet,grad_enb,averageGrad,averageSqGrad,epoch);

    % Display the training progress.
    if plots == "training-progress"
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        addpoints(lineLossTrain,epoch,double(gather(extractdata(loss_enb))))
        title("Epoch: " + epoch + ", Elapsed: " + string(D))
        drawnow
    end
end

%% Test
dlXTest = dlarray(xTest,'SSCB');
dlXTest = gpuArray(dlXTest);
YTest =single(yTest);

[dlYPred] = predict(dlnet,dlXTest);
[~,idx] = max(extractdata(dlYPred),[],1);
YPred = classes(idx);

[y_pred, scores] = classify(dlnet,dlXTest,YTest);


[y_pred, scores] = classify(dlnet,xTest,yTest);
prop(:,j) = scores(:,2);
[~,~,~,AUC(j)] = perfcurve(ans_auc(:,j),prop(:,j),0);
acc(j) = sum(double(y_pred == yTest)/numel(yTest));
% disp(sprintf('Test accuracy: %d %%',floor(acc*100)));
end
mean_acc = mean(acc);
disp(mean_acc)


mean_AUC = mean(AUC);
disp(mean_AUC)




