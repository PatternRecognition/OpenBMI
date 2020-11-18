clc; clear all; close all;
%% Session 2 - 2018
dd='G:\1_ConvertedData_brain_reaching';
filelist={'sub01_reaching_realMove'}; 
filelist2={'sub01_reaching_MI'};    
% 
%% Offline Analysis
for sub = 1: 1
%     tic
%     load('C:\Users\cvpr\Desktop\졸업준비자료\손목회전 각도실험\손목회전각도데이터\test\traj_template.mat');
    
    %% Train data
    [cnt, mrk, mnt]=eegfile_loadMatlab('G:\1_ConvertedData_brain_reaching\sub01_reaching_realMove');
    ival=[0 3000];
    
%     cnt.x=resample(cnt.x,4,5);
%     cnt.fs=80;
%     mrk.pos=ceil(mrk.pos*4/5);
%     mrk.fs=80;
%     
%     filtBank = {'[4 40]'};
%     cnt = proc_filtButter(cnt, 5, filtBank{1});
%     cnt = proc_commonAverageReference(cnt);
%     epo = cntToEpo(cnt,mrk,ival);
    
    band= [4 40];
    [b,a]= butter(5, band/cnt.fs*2);
    cnt_flt= proc_filt(cnt, b, a);

    epo = cntToEpo(cnt_flt, mrk, ival);
    
    
    epo.x=epo.x.*epo.x;
    epomove=proc_movingAverage(epo,100);    
    epo.x=log(epo.x);

    
    clear cnt mrk mnt
    
    %% Test 
    
    [cnt, mrk, mnt]=eegfile_loadMatlab('G:\1_ConvertedData_brain_reaching\sub01_reaching_MI');

    [b,a]= butter(5, band/cnt.fs*2);
    cnt_flt= proc_filt(cnt, b, a);

    epoimg = cntToEpo(cnt_flt, mrk, ival);
    
    
%     cnt.x=resample(cnt.x,4,5);
%     cnt.fs=80;
%     mrk.pos=ceil(mrk.pos*4/5);
%     mrk.fs=80;
%     
%     filtBank = {[4 40]};
%     cnt = proc_filtButter(cnt, 2, filtBank{1});
%     epoimg=cntToEpo(cnt, mrk, ival);
%     epoimg=proc_commonAverageReference(epoimg);
   
    
    epoimg.x=epoimg.x.*epoimg.x;
%     epoimg=proc_movingAverage(epoimg,100);
    epoimg.x=log(epoimg.x);
    
    epoimg.x=zscore(epoimg.x,1);
    
    clear cnt mrk mnt
   %% Data shuffling 
    randidx = randperm(size(epomove.y,2));
    for idx = 1: size(epomove.y,2)
        a.y(:,idx) = epomove.y(:,randidx(idx));
        a.x(:,:,idx) = epomove.x(:,:,randidx(idx));
    end
    epomove.y = a.y;
    epomove.x = a.x;
    
    %% Select Trial (Training and Test) - XTrain, XTest
    numtrainsize = 240;
    numvalsize = 60;
    numtestsize = 60;        
    
    
      %% Fair distribution for test dataset
    yOne = find(epomove.y(1,:) == 1); yOne = yOne(1: numtestsize/4);
    yTwo = find(epomove.y(2,:) == 1); yTwo = yTwo(1:numtestsize/4);
    yThree = find(epomove.y(3,:) == 1); yThree = yThree(1:numtestsize/4);
    yFour = find(epomove.y(4,:) == 1); yFour = yFour(1:numtestsize/4);
    
    epovalLabel = sort([yOne yTwo yThree yFour]);
    epoval = proc_selectEpochs(epomove, epovalLabel);
    epo= proc_selectEpochs(epomove, 'not', epovalLabel);
    
    traj_template=cat(1,zeros(80,4),traj_template);
%      traj=traj_template;
%      traj=traj;
%      traj=[[0 0 0 0]; traj];
%      
%      
%      traj_vel=80*diff(traj);
%      traj_vel=abs(traj_vel);
%      clear traj traj_template
%      
%      traj_template=traj_vel;
    
   %%
    XTrain = cell(numtrainsize, 1);
    for xtr = 1: numtrainsize
        XTrain{xtr} = epo.x(:,:,xtr)';
    end
   %% Data Construction for regression - YTrain
    YTrain = cell(numtrainsize, 1);
    for ytr = 1: size(epo.y,2)
        b = find(epo.y(:,ytr)==1);
        YTrain{ytr} = traj_template(1:321,b)';
    end
    
   % clear yOne yTwo yThree yFour     
   
   %%
    XVal = cell(numvalsize, 1);
    for xval = 1: numvalsize
        XVal{xval} = epoval.x(:,:,xval)';
    end
   %% Data Construction for regression - YTrain
    YVal = cell(numvalsize, 1);
    for yval = 1: size(epoval.y,2)
        b = find(epoval.y(:,yval)==1);
        YVal{yval} = traj_template(1:321,b)';
    end
   
   clear yOne yTwo yThree yFour a
    
   %% Data shuffling 
    randidx = randperm(size(epoimg.y,2));
    for idx = 1: size(epoimg.y,2)
        a.y(:,idx) = epoimg.y(:,randidx(idx));
        a.x(:,:,idx) = epoimg.x(:,:,randidx(idx));
    end
    epoimg.y = a.y;
    epoimg.x = a.x;
    
   %% Fair distribution for test dataset
    yOne = find(epoimg.y(1,:) == 1); 
    yOne = yOne(1: numtestsize/4);
    yTwo = find(epoimg.y(2,:) == 1); 
    yTwo = yTwo(1:numtestsize/4);
    yThree = find(epoimg.y(3,:) == 1); 
    yThree = yThree(1:numtestsize/4);
    yFour = find(epoimg.y(4,:) == 1); 
    yFour = yFour(1:numtestsize/4);
    
    epotestLabel = sort([yOne yTwo yThree yFour]);
    epoimg = proc_selectEpochs(epoimg, epotestLabel);
    epointra= proc_selectEpochs(epoimg, 'not', epotestLabel);
    
       
   %%
    XImg = cell(numtrainsize, 1);
    for xtr = 1: numtrainsize
        XImg{xtr} = epointra.x(:,:,xtr)';
    end
   %% Data Construction for regression - YTrain
    YImg = cell(numtrainsize, 1);
    for ytr = 1: size(epointra.y,2)
        b = find(epointra.y(:,ytr)==1);
        YImg{ytr} = traj_template(1:321,b)';
    end
   
    
   %%
    XTest = cell(numtestsize, 1);
    for xtst = 1: numtestsize
        XTest{xtst} = epoimg.x(:,:,xtst)';
    end
    %% Data Construction for regression - YTrain, YTest
    YTest = cell(numtestsize, 1);
    
    for ytst = 1: size(epoimg.y,2)
        b = find(epoimg.y(:,ytst)==1);
        YTest{ytst} = traj_template(1:321,b)';
    end
    
    clear yOne yTwo yThree yFour 
    
   
    
    %% Deep Learning
    numResponses = size(YTrain{1},1);
    featureDimension = size(XTrain{1},1);
    numHiddenUnits = 32;
    
    layers = [ ...
        sequenceInputLayer(featureDimension)
        lstmLayer(numHiddenUnits,'Name','bilstm1')
        lstmLayer(numHiddenUnits,'Name','bilstm2')
%        lstmLayer(numHiddenUnits,'Name','bilstm3')
%         lstmLayer(numHiddenUnits,'Name','bilstm4')
%         lstmLayer(numHiddenUnits,'Name','bilstm5')
%         dropoutLayer(0.2)
        fullyConnectedLayer(numResponses)
        regressionLayer];
    
    maxEpochs = 300;
    miniBatchSize = 40;
    
    %     validationData = {XTest, YTest};
    options = trainingOptions('adam', ...
        'ExecutionEnvironment','gpu',...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize',miniBatchSize, ...
        'InitialLearnRate',0.01, ...
        'GradientThreshold', 1,...
        'GradientThresholdMethod','global-l2norm',...
        'Shuffle','never', ...
        'Plots','training-progress',...
        'Verbose',0);
    
    transfer_net = trainNetwork(XTrain,YTrain,layers,options);
    intra_net = trainNetwork(XImg,YImg,layers,options);
    
    YPred_intra = predict(intra_net,XTest,'MiniBatchSize',miniBatchSize);
    YPred_Test = predict(transfer_net,XTest,'MiniBatchSize',miniBatchSize);
    
    for k = 1:size(YTest,1)
        YTest2((size(epoimg.x,1)*(k-1))+1:size(epoimg.x,1)*k) = YTest{k};
        YPred2((size(epoimg.x,1)*(k-1))+1:size(epoimg.x,1)*k) = YPred_intra{k};
    end
    
    for k = 1:size(YTest,1)
%         YTest3((size(epotest.x,1)*(k-1))+1:size(epotest.x,1)*k) = YTest{k};
        YPred3((size(epoimg.x,1)*(k-1))+1:size(epoimg.x,1)*k) = YPred_Test{k};
    end
    
    figure(sub)
%     subplot(2,1,1);
%     plot(-YPred2(1:3000) , 'r');hold on; plot(-YTest2(1:3000) , 'b','Linestyle' , '--');  grid on;
    %xticks(400:400:4800);
    %xticklabels({'-3\pi','-2\pi','-\pi','0','\pi','2\pi','3\pi'})
     subplot(2,1,1);
    plot(-YPred2(1:3000) , 'r');hold on; plot(-YTest2(1:3000) , 'b','Linestyle' , '--');  grid on;
    ylim([-0.5 3]);
    subplot(2,1,2);    
    plot(-YPred3(1:3000) , 'r');hold on; plot(-YTest2(1:3000) , 'b','Linestyle' , '--');  grid on;
        %xticks(400:400:4800);
    ylim([-0.5 3]);
    
    nrmse(sub,1) = sqrt(mean((YPred2-YTest2).^2));
    nrmse(sub,2) = sqrt(mean((YPred3-YTest2).^2));
    cc_temp=corrcoef(YPred2(:),YTest2(:)); 
    cc(sub,1) = cc_temp(1,2);
    cc(sub,1)
    clear cc_temp
    cc_temp=corrcoef(YPred3(:),YTest2(:)); 
    cc(sub,2) = cc_temp(1,2);
    cc(sub,2)
    %nrmse(sub)

end

%%

