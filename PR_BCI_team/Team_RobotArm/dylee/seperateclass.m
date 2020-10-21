%% @ written by Kyung-Hwan Shim
% Construction Date: 20190620

%% Initializing
clear all;
close all;
clc;

%% file directory

dd = 'C:\Users\HANSEUL\Desktop\AAAI\GIGA_20190708_jmlee/';
twist='GIGA_20190715_jmlee_twist_MI';
reaching='GIGA_20190715_jmlee_reaching_MI';
grasp='GIGA_20190715_jmlee_multigrasp_MI';


% dd = 'C:\Users\HANSEUL\Desktop\AAAI\GIGA_20190710_dslim/';
% twist='GIGA_20190717_dslim_twist_MI';
% reaching='GIGA_20190717_dslim_reaching_MI';
% grasp='GIGA_20190717_dslim_multigrasp_MI';


% dd = 'C:\Users\HANSEUL\Desktop\AAAI\GIGA_20190712_wjyoon/';
% twist='GIGA_20190719_wjyoon_twist_MI';
% reaching='GIGA_20190719_wjyoon_reaching_MI';
% grasp='GIGA_20190719_wjyoon_multigrasp_MI';




fsIdx=1;
fs={'100','250','1000'};

ref='FCz';
channelMatrix={'F3','F1','Fz','F2','F4';
    'FC3','FC1','FCz','FC2','FC4';
    'C3','C1', 'Cz', 'C2', 'C4';
    'CP3','CP1','CPz','CP2','CP4';
    'P3','P1','Pz','P2','P4'};

trainRatio=0.8;
modIdx=1;
modality={'MI','realMove'};

% for sub=1:length(filelist)
ival=[0 3000];
[cntGrasp,mrkGrasp,mntGrasp]=eegfile_loadMatlab([dd grasp]);
[cntTwist,mrkTwist,mntTwist]=eegfile_loadMatlab([dd twist]);
[cntReach,mrkReach,mntReach]=eegfile_loadMatlab([dd reaching]);

cntGrasp=proc_filtButter(cntGrasp,5,[4 40]);
cntTwist=proc_filtButter(cntTwist,5,[4 40]);
cntReach=proc_filtButter(cntReach,5,[4 40]);

epoGrasp=cntToEpo(cntGrasp,mrkGrasp,ival);
epoTwist=cntToEpo(cntTwist,mrkTwist,ival);

% Reaching의 left와 right 명칭 겹침 -> Twist에서 미리 셋팅
epoTwist.className = {'Twist_Left', 'Twist_Right', 'Rest'};
epoReach=cntToEpo(cntReach,mrkReach,ival);

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
epoReach=proc_selectChannels(epoReach,{'F3','F1','Fz','F2','F4',...
    'FC3','FC1','FCz','FC2','FC4',...
    'C3','C1', 'Cz', 'C2', 'C4', ...
    'CP3','CP1','CPz','CP2','CP4',...
    'P3','P1','Pz','P2','P4'});

%% Retrieve Trial amounts

% Grasp - 4 classes (Cylindrical, Spherical, Lumbrical, Rest)
epoCylindrical=proc_selectClasses(epoGrasp,{'Cylindrical'});
epoSpherical=proc_selectClasses(epoGrasp,{'Spherical'});
epoLumbrical=proc_selectClasses(epoGrasp,{'Lumbrical'});
epoGrasp_rest=proc_selectClasses(epoGrasp,{'Rest'});

epo_Grasp = proc_appendEpochs(epoCylindrical,epoSpherical);
epo_Grasp = proc_appendEpochs(epo_Grasp,epoLumbrical);
epo_Grasp.className = {'Grasping'};
epo_Grasp.y = ones(1,size(epo_Grasp.y,2));

% Twist - 3 classes (Right, Left, Rest)
epoTwist_Right=proc_selectClasses(epoTwist,{'Twist_Right'});
epoTwist_Left=proc_selectClasses(epoTwist,{'Twist_Left'});
epoTwist_Rest=proc_selectClasses(epoTwist,{'Rest'});

epo_Twist = proc_appendEpochs(epoTwist_Right, epoTwist_Left);
epo_Twist.className = {'Twisting'};
epo_Twist.y = ones(1, size(epo_Twist.y,2));

% Reach - 7 classes (Left, Right, Forward, Bachward, Up, Down, Rest)
epoLeft=proc_selectClasses(epoReach,{'Left'});
epoRight=proc_selectClasses(epoReach,{'Right'});
epoForward=proc_selectClasses(epoReach,{'Forward'});
epoBackward=proc_selectClasses(epoReach,{'Backward'});
epoUp=proc_selectClasses(epoReach,{'Up'});
epoDown=proc_selectClasses(epoReach,{'Down'});
epoReaching_Rest=proc_selectClasses(epoReach,{'Rest'});

epo_Reaching = proc_appendEpochs(epoLeft, epoRight);
epo_Reaching = proc_appendEpochs(epo_Reaching, epoForward);
epo_Reaching = proc_appendEpochs(epo_Reaching, epoBackward);
epo_Reaching = proc_appendEpochs(epo_Reaching, epoUp);
epo_Reaching = proc_appendEpochs(epo_Reaching, epoDown);

epo_Reaching.className = {'Reaching'};
epo_Reaching.y = ones(1, size(epo_Reaching.y,2));

% Rest
epoRest= proc_appendEpochs(epoTwist_Rest,epoReaching_Rest);
epoRest= proc_appendEpochs(epoRest,epoGrasp_rest);
epoTwist_Rest=proc_selectClasses(epoTwist,{'Rest'});
epoRest.y = ones(1, size(epoRest.y,2));


%% Data shuffling

epo_Reaching.x = datasample(epo_Reaching.x,size(epo_Reaching.x,3),3, 'Replace', false);
epo_Twist.x = datasample(epo_Twist.x , size(epo_Twist.x,3),3, 'Replace', false);
epo_Grasp.x = datasample(epo_Grasp.x , size(epo_Grasp.x,3),3, 'Replace', false);
epoRest.x = datasample(epoRest.x , size(epo_Grasp.x,3),3, 'Replace', false);

%% Setting training data

minTrial=min([size(epo_Reaching.x,3),size(epo_Twist.x,3),size(epo_Grasp.x,3)]);
trainNum=minTrial*trainRatio;

%%
epo_ReachingTrain=proc_selectEpochs(epo_Reaching,1:trainNum);
epo_TwistTrain=proc_selectEpochs(epo_Twist,1:trainNum);
epo_GraspTrain=proc_selectEpochs(epo_Grasp,1:trainNum);
epoRestTrain=proc_selectEpochs(epoRest,1:trainNum);

%%
epo_ReachingTest=proc_selectEpochs(epo_Reaching,trainNum+1:size(epo_Reaching.x,3));
epo_TwistTest=proc_selectEpochs(epo_Twist,trainNum+1:size(epo_Twist.x,3));
epo_GraspTest=proc_selectEpochs(epo_Grasp,trainNum+1:size(epo_Grasp.x,3));
epoRestTest=proc_selectEpochs(epoRest,trainNum+1:size(epoRest.x,3));
%% Concat
epoTrain=proc_appendEpochs(epo_ReachingTrain,epo_TwistTrain);
epoTrain=proc_appendEpochs(epoTrain,epo_GraspTrain);
epoTrain=proc_appendEpochs(epoTrain,epoRestTrain);
Train=epoTrain;

epoTest=proc_appendEpochs(epo_ReachingTest,epo_TwistTest);
epoTest=proc_appendEpochs(epoTest,epo_GraspTest);
epoTest=proc_appendEpochs(epoTest,epoRestTest);
Test=epoTest;

%     epoTest=proc_appendEpochs(epoCylindricalTest,epoForwardTest);
%     epoTest=proc_appendEpochs(epoTest,epoTwist_RightTest);
%     epoTest=proc_appendEpochs(epoTest,epoRestTest);

trainX=epoTrain.x;testX=epoTest.x;
trainY=epoTrain.y;testY=epoTest.y;
trainY=vec2ind(trainY)-1;
testY=vec2ind(testY)-1;
save(["C:\Users\HANSEUL\Desktop\AAAI\3class\jmlee/"+fs{fsIdx}+"/train/sub_3class"],'trainX','trainY');
save(["C:\Users\HANSEUL\Desktop\AAAI\3class\jmlee/"+fs{fsIdx}+"/test/sub_3class"],'testX','testY');