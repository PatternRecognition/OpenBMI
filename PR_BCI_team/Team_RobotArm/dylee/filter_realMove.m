% Construction Data_MI
% MI: 50 trials
% 4-class (Cylindrical/Spherical/LeftTwist/RightTwist)
%% Initializing
clc; close all; clear all;
%% file
dd='C:\Users\Doyeunlee\Desktop\Journal_dylee\1_ConvertedData\';
fsldx=2;
fs={'100','250','1000'};
ref='FCz';
channelMatrix={'F3','F1','Fz','F2','F4';
    'FC3','FC1','FCz','FC2','FC4';
    'C3','C1', 'Cz', 'C2', 'C4';
    'CP3','CP1','CPz','CP2','CP4';
    'P3','P1','Pz','P2','P4'};

trainRatio=0.8;
modldx=1;
modality={'MI','realMove'};

filelist={'sub01','sub02','sub03','sub04','sub05'};

%%
for sub = 1:length(filelist)
    ival = [0 3000];
    
    % Data load - MI
    [cntGrasp,mrkGrasp,mntGrasp]=eegfile_loadMatlab([dd 'realMove' '\' fs{fsldx} '\' filelist{sub} '_multigrasp_' 'realMove']);
    [cntTwist,mrkTwist,mntTwist]=eegfile_loadMatlab([dd 'realMove' '\' fs{fsldx} '\' filelist{sub} '_twist_' 'realMove']);
    
    % theta, mu, and beta band
    filterBank = {'[4 8]','[8 13]','[13 40]'};
    
    for filt = 1:length(filterBank)
        filtBank = {filt};
        
        % Butterworth
        cntGrasp=proc_filtButter(cntGrasp,5,[4 40]);
        cntTwist=proc_filtButter(cntTwist,5,[4 40]);
        
        % CAR
        cntGrasp = proc_commonAverageReference(cntGrasp);
        cntTwist = proc_commonAverageReference(cntTwist);
        
        epoGrasp=cntToEpo(cntGrasp,mrkGrasp,ival);
        epoTwist=cntToEpo(cntTwist,mrkTwist,ival);
        
        epoGrasp=proc_selectChannels(epoGrasp,{'F3','F1','Fz','F2','F4',...
            'FC3','FC1','FCz','FC2','FC4',...
            'C3','C1', 'Cz', 'C2', 'C4', ...
            'CP3','CP1','CPz','CP2','CP4',...
            'P3','P1','Pz','P2','P4'});
        epoTwist=proc_selectChannels(epoTwist,{'F3','F1','Fz','F2','F4',...
            'FC3','FC1','FCz','FC2','FC4',...
            'C3','C1', 'Cz', 'C2', 'C4', ...
            'CP3','CP1','CPz','CP2','CP4',...
            'P3','P1','Pz','P2','P4'});
        
        
        % Retrieve trial amounts
        %% 4-class: Forward, Cylindrical, LeftTwist, Rest
        epoCylindrical=proc_selectClasses(epoGrasp,{'Cylindrical'});
        epoSpherical=proc_selectClasses(epoGrasp,{'Spherical'});
        epoLeftTwist=proc_selectClasses(epoTwist,{'LeftTwist'});
        epoRightTwist=proc_selectClasses(epoTwist,{'RightTwist'});
        
        % append
        epo = proc_appendEpochs(epoCylindrical,epoSpherical);
        epo = proc_appendEpochs(epo,epoLeftTwist);
        epo = proc_appendEpochs(epo,epoRightTwist);
        
        % Data shuffling
        epoCylindrical.x=datasample(epoCylindrical.x,size(epoCylindrical.x,3),3,'Replace',false);
        epoSpherical.x=datasample(epoSpherical.x,size(epoSpherical.x,3),3,'Replace',false);
        epoLeftTwist.x=datasample(epoLeftTwist.x,size(epoLeftTwist.x,3),3,'Replace',false);
        epoRightTwist.x=datasample(epoRightTwist.x,size(epoRightTwist.x,3),3,'Replace',false);
        
        % CSP 
        [csp_fv,csp_w,csp_eig]=proc_multicsp(epo,3);
        proc=struct('memo','csp_w');
        
        proc.train=['[fv,csp_w]= proc_multicsp(fv, 3);' ...
            'fv=proc_variance(fv);' ...
            'fv=proc_logarithm(fv);'];
        
        proc.apply=['fv=proc_linearDerivation(fv, csp_w);','fv=proc_variance(fv);','fv=proc_logarithm(fv);'];
               
       
        % Setting training Data
        minTrial=min([size(epoCylindrical.x,3), ...
            size(epoSpherical.x,3), ...
            size(epoLeftTwist.x,3), size(epoRightTwist.x,3)]);
        trainNum=minTrial*trainRatio;
        
        % training
        epoCylindricalTrain=proc_selectEpochs(epoCylindrical,1:trainNum);
        epoSphericalTrain=proc_selectEpochs(epoSpherical,1:trainNum);
        epoLeftTwistTrain=proc_selectEpochs(epoLeftTwist,1:trainNum);
        epoRightTwistTrain=proc_selectEpochs(epoRightTwist,1:trainNum);
        
        % test
        epoCylindricalTest=proc_selectEpochs(epoCylindrical,trainNum+1:size(epoCylindrical.x,3));
        epoSphericalTest=proc_selectEpochs(epoSpherical,trainNum+1:size(epoSpherical.x,3));
        epoLeftTwistTest=proc_selectEpochs(epoLeftTwist,trainNum+1:size(epoLeftTwist.x,3));
        epoRightTwistTest=proc_selectEpochs(epoRightTwist,trainNum+1:size(epoRightTwist.x,3));
        
        % Concat
        epoTrain=proc_appendEpochs(epoCylindricalTrain,epoSphericalTrain);
        epoTrain=proc_appendEpochs(epoTrain,epoLeftTwistTrain);
        epoTrain=proc_appendEpochs(epoTrain,epoRightTwistTrain);
        Train=epoTrain;
        
        epoTest=proc_appendEpochs(epoCylindricalTest,epoSphericalTest);
        epoTest=proc_appendEpochs(epoTest,epoLeftTwistTest);
        epoTest=proc_appendEpochs(epoTest,epoRightTwistTest);
        Test=epoTest;
        
        
        %%
        trainX=epoTrain.x;
        testX=epoTest.x;
        trainY=epoTrain.y;testY=epoTest.y;
        trainY=vec2ind(trainY)-1;
        testY=vec2ind(testY)-1;
        str= char('_train');
        str2=char('_test');
        str3=char('.mat');
%         str4=char();
        save(['C:\Users\Doyeunlee\Desktop\AAAI_20190717\Dataconstruction_filter\realMove_train/',num2str(sub),filterBank{filt},str3],'trainX','trainY');
        save(['C:\Users\Doyeunlee\Desktop\AAAI_20190717\Dataconstruction_filter\realMove_test/',num2str(sub),filterBank{filt},str3],'testX','testY');
    end % filterBank
end % subject

disp('Done');


