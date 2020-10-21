clc; clear all; close all;
%% Session 2 - 2018
%% Load the EEG data
% dd = 'D:\robotarm\Robot arm project\Reach\';
%% Load the EEG data
dd = 'C:\Users\BHLee\Desktop\practice\data\MI_matfile/';
%% Real move
%filelist = {'20180208_jgyoon_reaching_realMove', '20180219_bwyu_reaching_realMove', '20180226_dkhan_reaching_realMove', '20180228_sjahn_reaching_realMove',...
%    '20180305_bwjeong_reaching_realMove', '20180206_msyun_reaching_realMove', '20180222_shchoi_reaching_realMove', '20180410_chsong_reaching_realMove',...
%   '20180517_stoh_reaching_realMove', '20180605_msoh_reaching_realMove','20190312_bhkwon_reaching_realMove'};
%% Motor Imagery
filelist= {'20180208_jgyoon_reaching_MI','20180219_bwyu_reaching_MI','20180226_dkhan_reaching_MI', '20180228_sjahn_reaching_MI','20180305_bwjeong_reaching_MI',...
    '20180206_msyun_reaching_MI', '20180222_shchoi_reaching_MI', '20180410_chsong_reaching_MI',...
    '20180517_stoh_reaching_MI','20180605_msoh_reaching_MI','20190312_bhkwon_reaching_MI'};
%% Motor Imagery
%  filelist= {'subject1','subject2', 'subject3', 'subject4', 'subject5', 'subject6','subject7', ...
%      'subject8', 'subject9', 'subject10', 'subject11', 'subject12', 'subject13', 'subject14','subject15' };
%% Single Subject
% filelist = {'Reach20190312_bhkwon_reaching_MI','Reach20190312_bhkwon_reaching_realMove'};

%% Parameter setting
trainRatio=0.5;
numClass = 6;
task = 1;

%%
for sub =1:length(filelist)
    sub
    disp('Refine....');
    %% Load motion data
    load('D:\robotarm\Robot arm project\Mat converting\modiedGroundTruth');
    re_velocity = re_velocity';
    re_velo = cell(6,1);
    for kk=1: size(re_velocity,1)
        re_velo{kk} = [re_velocity{kk,1} re_velocity{kk,2} re_velocity{kk,3}];
    end
    
    %% Parameter setting
    ival = [0 4000];
    %% EEG data convert
    [cnt, mrk, mnt]=eegfile_loadMatlab([dd filelist{sub}]);
    %% Band-pass filtering
    % IIR filter (Butterworth)
    filterBank = {'[0.1 40]', '[0.1 1]', '[4 7]', '[8 15]', '[16 30]', '[8 30]'};
    %     filterBank = {'[4 40]'};
    for filt=1:length(filterBank)
        filtBank = {filt};
        
        
        cnt = proc_filtButter(cnt, 5, filtBank{1});
        
        %% Normalized EEG signals
        %     cnt = proc_normalize(cnt);
        %% Spatial Filtering
        cnt = proc_commonAverageReference(cnt);
        %% Data segmentation
        epo = cntToEpo(cnt, mrk, ival);
        %% Select classes (6 classes)
        %     epo = proc_selectClasses(epo, {'Forward', 'Backward', 'Left', 'Right'});
        epo = proc_selectClasses(epo, {'Forward', 'Backward', 'Left', 'Right', 'Up', 'Down'});
        %         epo = proc_selectClasses(epo, {'Forward', 'Backward', 'Left', 'Right'});
        %     epo = proc_selectClasses(epo, {'elbow_flexion', 'elbow_extension', 'forearm_supination', 'forearm_pronation', 'hand_open', 'hand_close'});
        
        %% Select channels
        epo = proc_selectChannels(epo, {'FC5','FC3','FC1','FC2','FC4','FC6',...
            'C5','C3','C1', 'Cz', 'C2', 'C4', 'C6',...
            'CP5','CP3','CP1','CPz','CP2','CP4','CP6'});
        
        epoLeft=proc_selectClasses(epo,{'Left'});
        epoRight=proc_selectClasses(epo,{'Right'});
        epoForward=proc_selectClasses(epo,{'Forward'});
        epoBackward=proc_selectClasses(epo,{'Backward'});
        
        epoUp=proc_selectClasses(epo,{'Up'});
        epoDown=proc_selectClasses(epo,{'Down'});
        
        %% data shuffling
        epoLeft.x=datasample(epoLeft.x,size(epoLeft.x,3),3,'Replace',false);
        epoRight.x=datasample(epoRight.x,size(epoRight.x,3),3,'Replace',false);
        epoForward.x=datasample(epoForward.x,size(epoForward.x,3),3,'Replace',false);
        epoBackward.x=datasample(epoBackward.x,size(epoBackward.x,3),3,'Replace',false);
        epoUp.x=datasample(epoUp.x,size(epoUp.x,3),3,'Replace',false);
        epoDown.x=datasample(epoDown.x,size(epoDown.x,3),3,'Replace',false);
        %% train test split
        trainNum=round(size(epoLeft.x,3)*trainRatio);
        
        epoLeftTrain=proc_selectEpochs(epoLeft,1:trainNum);
        epoRightTrain=proc_selectEpochs(epoRight,1:trainNum);
        epoForwardTrain=proc_selectEpochs(epoForward,1:trainNum);
        epoBackwardTrain=proc_selectEpochs(epoBackward,1:trainNum);
        
        epoUpTrain=proc_selectEpochs(epoUp,1:trainNum);
        epoDownTrain=proc_selectEpochs(epoDown,1:trainNum);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        epoLeftTest=proc_selectEpochs(epoLeft,trainNum+1:size(epoLeft.x,3));
        epoRightTest=proc_selectEpochs(epoRight,trainNum+1:size(epoRight.x,3));
        epoForwardTest=proc_selectEpochs(epoForward,trainNum+1:size(epoForward.x,3));
        epoBackwardTest=proc_selectEpochs(epoBackward,trainNum+1:size(epoBackward.x,3));
        
        epoUpTest=proc_selectEpochs(epoUp,trainNum+1:size(epoUp.x,3));
        epoDownTest=proc_selectEpochs(epoDown,trainNum+1:size(epoDown.x,3));
        
        %% Data Construction for Training
        trainLR=proc_appendEpochs(epoLeftTrain, epoRightTrain); trainFB=proc_appendEpochs(epoForwardTrain, epoBackwardTrain);
        trainUD=proc_appendEpochs(epoUpTrain, epoDownTrain);
        trainLRFB=proc_appendEpochs(trainLR,trainFB);
        trainLRFBUD = proc_appendEpochs(trainLRFB,trainUD);
        
        switch(numClass)
            case 4
                %% %%%%%%%%% 4-class %%%%%%%%%%%
                trainX=trainLRFB.x;
                for ytr = 1: size(trainLRFB.y,2)
                    b = find(trainLRFB.y(:,ytr)==1);
                    trainY(:,:,ytr) = re_velo{b,1};
                end
            case 6
                %% %%%%%%%%% 6-class %%%%%%%%%%%
                mat = dir('*.mat');
                for q = 1:length(mat) load(mat(q).name)
                end
                trainX=trainLRFBUD.x;
                for ytr = 1: size(trainLRFBUD.y,2)
                    b = find(trainLRFBUD.y(:,ytr)==1);
                    trainY(:,:,ytr) = re_velo{b,1};
                end
                
                
        end
        
        %% Data Construction for Testing
        testLR=proc_appendEpochs(epoLeftTest, epoRightTest); testFB = proc_appendEpochs(epoForwardTest, epoBackwardTest);
        testUD=proc_appendEpochs(epoUpTest, epoDownTest);
        testLRFB=proc_appendEpochs(testLR,testFB);
        testLRFBUD = proc_appendEpochs(testLRFB, testUD);
        
        switch(numClass)
            %%%%%%%%%%% 4-class %%%%%%%%%%%
            case 4
                testX=testLRFB.x;
                for ytst = 1: size(testLRFB.y,2)
                    b = find(testLRFB.y(:,ytst)==1);
                    testY(:,:,ytst) = re_velo{b,1};
                end
                %%%%%%%%%% Save the .mat %%%%%%%%%%
                if task == 1
                    save(['D:\Robot arm project\results'  (sub)], 'trainX', 'trainY');
                    save(['D:\Robot arm project\results' num2str(sub)], 'testX','testY');
                else
                    save(['D:\Robot arm project\results' num2str(sub)], 'trainX', 'trainY');
                    save(['D:\Robot arm project\results' num2str(sub)], 'testX','testY');
                end
                %%%%%%%%%%% 6-class %%%%%%%%%%%
            case 6
                testX=testLRFBUD.x;
                for ytst = 1: size(testLRFBUD.y,2)
                    b = find(testLRFBUD.y(:,ytst)==1);
                    testY(:,:,ytst) = re_velo{b,1};
                end
                %%%%%%%%%% Save the .mat %%%%%%%%%%
                
                str= char('_train');
                str2=char('_test');
                str3=char('.mat');
                %%%%%%%%%% 에러메세지 생략%%%%%%%%%%%%%
                %  w = warning('query','last');
                %   id = w.identifier;
                %    warning('off',id)
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                %% ME
                %            mkdir (['./ME/train/',filterBank{filt}]);
                %            mkdir (['./ME/test/',filterBank{filt}]);
                %
                %            if  task == 1
                %              save(['./ME/train/',filterBank{filt}, './sub', num2str(sub),str3], 'trainX', 'trainY');
                %              save(['./ME/test/',filterBank{filt}, './sub', num2str(sub),str3], 'testX', 'testY');
                %
                %           else
                %               save(['./ME/train/',filterBank{filt}, './sub', num2str(sub),str3], 'trainX', 'trainY');
                %               save(['./ME/test/',filterBank{filt}, './sub', num2str(sub),str3], 'testX', 'testY');
                %              end
                
                %% MI
                %             mkdir (['./MI/train/', filterBank{filt}]);
                %             mkdir (['./MI/test/', filterBank{filt}]);
                %                if  task == 1
                %                       save(['./MI/train/',filterBank{filt}, './sub', num2str(sub),str3], 'trainX', 'trainY');
                %                     save(['./MI/test/',filterBank{filt}, './sub', num2str(sub),str3], 'testX', 'testY');
                %               else
                %                  save(['./MI/train/',filterBank{filt}, './sub', num2str(sub),str3], 'trainX', 'trainY');
                %                  save(['./MI/test/',filterBank{filt}, './sub', num2str(sub),str3], 'testX', 'testY');
                %                end
                %% MI
                mkdir (['C:\Users\BHLee\Desktop\practice\data\MI_matfile/train/', filterBank{filt}]);
                mkdir (['C:\Users\BHLee\Desktop\practice\data\MI_matfile/test/', filterBank{filt}]);
                if  task == 1
                    save(['./MI/train/',filterBank{filt}, './sub', num2str(sub),str3], 'trainX', 'trainY');
                    save(['./MI/test/',filterBank{filt}, './sub', num2str(sub),str3], 'testX', 'testY');
                else
                    save(['./MI/train/',filterBank{filt}, './sub', num2str(sub),str3], 'trainX', 'trainY');
                    save(['./MI/test/',filterBank{filt}, './sub', num2str(sub),str3], 'testX', 'testY');
                end
        end
        
    end
    %%
    clear trainX trainY testX testY;
    disp('Save Done....');
    
end