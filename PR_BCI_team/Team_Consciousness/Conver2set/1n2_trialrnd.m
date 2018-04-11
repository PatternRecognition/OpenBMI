% 1st step: trial randomization in the same subject
% -----------------------------------------------------------------------------------
% 1-1) struct -> arracy
% 1-2) modifying the length of epoch: 10sec for each class -> 5 sec
%      this step is aim to double the number of UCS trials
% 1-3) WFN LOC UCS ROC REV randomization -> extracting the same number of
%      trials for each class
% 1-4) merge 5 states to 3 classes
% -----------------------------------------------------------------------------------

clc; clear; close;

fpath=('D:\anedata_mat\');

for sub=1:10
    
    fname_WFN=(sprintf('MM_WFN_%d.mat', sub));
    fname_LOC=(sprintf('MM_LOC_%d.mat', sub));
    fname_UCS=(sprintf('MM_UCS_%d.mat', sub));
    fname_ROC=(sprintf('MM_ROC_%d.mat', sub));
    fname_REV=(sprintf('MM_REV_%d.mat', sub));
    
    file_WFN=fullfile(fpath, fname_WFN);
    file_LOC=fullfile(fpath, fname_LOC);
    file_UCS=fullfile(fpath, fname_UCS);
    file_ROC=fullfile(fpath, fname_ROC);
    file_REV=fullfile(fpath, fname_REV);
        
    %% Struct -> Array
    MM_WFN=load(file_WFN);
    MM_LOC=load(file_LOC);
    MM_UCS=load(file_UCS);
    MM_ROC=load(file_ROC);
    MM_REV=load(file_REV);
   
    c_MM_WFN=struct2cell(MM_WFN);
    c_MM_LOC=struct2cell(MM_LOC);
    c_MM_UCS=struct2cell(MM_UCS);
    c_MM_ROC=struct2cell(MM_ROC);
    c_MM_REV=struct2cell(MM_REV);
    
    arr_MM_WFN=c_MM_WFN{1};
    arr_MM_LOC=c_MM_LOC{1};
    arr_MM_UCS=c_MM_UCS{1};
    arr_MM_ROC=c_MM_ROC{1};
    arr_MM_REV=c_MM_REV{1};
    
    %% modifying the length of epoch (5sec)
    % WFN 1~1000
    epo_MM_WFN = arr_MM_WFN(:, (1:1000), :);
    
    % LOC 501~1500
    epo_MM_LOC = arr_MM_LOC(:, (501:1500), :);
    
    % UCS 1~1000
    epo1_MM_UCS = arr_MM_UCS(:, (1:1000), :);
    
    % UCS (2) 1001~2000
    epo2_MM_UCS = arr_MM_UCS(:, (1001:2000), :);

    % ROC 501~1500
    epo_MM_ROC = arr_MM_ROC(:, (501:1500), :);
    
    % REV 1~1000
    epo_MM_REV = arr_MM_REV(:, (1:1000), :);
    
    %% concatenating UCS epoch 1 and 2
    epo_MM_UCS = cat(3, epo1_MM_UCS, epo2_MM_UCS);
    
    %% WFN randomization
    % extracting 2 WFN trials randomly -> merge them
    MM_WFN_1 = epo_MM_WFN(:, :, randperm(29, 1));
    MM_WFN_2 = epo_MM_WFN(:, :, randperm(29, 1));

    MM_WFN_rnd = cat(3, MM_WFN_1, MM_WFN_2);    
    
    %% LOC randomization
    % extracting 2 LOC trials randomly -> merge them
    MM_LOC_1 = epo_MM_LOC(:, :, randperm(3, 1));
    MM_LOC_2 = epo_MM_LOC(:, :, randperm(3, 1));
    
    MM_LOC_rnd = cat(3, MM_LOC_1, MM_LOC_2);  
    
    %% UCS randomization
    % extracting 4 UCS trials randomly -> merge them 
    MM_UCS_1 = epo_MM_UCS(:, :, randperm(6, 1));
    MM_UCS_2 = epo_MM_UCS(:, :, randperm(6, 1));
    MM_UCS_3 = epo_MM_UCS(:, :, randperm(6, 1));
    MM_UCS_4 = epo_MM_UCS(:, :, randperm(6, 1));
   
    MM_UCS_rnd = cat(3, MM_UCS_1, MM_UCS_2, MM_UCS_3, MM_UCS_4);
    
    %% ROC randomization
    % extracting 2 ROC trials randomly -> merge them
    MM_ROC_1 = epo_MM_ROC(:, :, randperm(3, 1));
    MM_ROC_2 = epo_MM_ROC(:, :, randperm(3, 1));
   
    MM_ROC_rnd = cat(3, MM_ROC_1, MM_ROC_2);  
    
    %% REV randomization
    % extracting 2 REV trials randomly -> merge them
    MM_REV_1 = epo_MM_REV(:, :, randperm(30, 1));
    MM_REV_2 = epo_MM_REV(:, :, randperm(30, 1));

    MM_REV_rnd = cat(3, MM_REV_1, MM_REV_2);
    
       %% concatenating 5 states to CS / US / TR
    MM_CS = cat(3, MM_WFN_rnd, MM_REV_rnd);
    MM_US = MM_UCS_rnd;
    MM_TR = cat(3, MM_LOC_rnd, MM_ROC_rnd);
    
    %% concatenating three classes
    % concatenating CS/US/TR
    MM_10_order = cat(3, MM_CS, MM_US, MM_TR);
    
    % trial numbering: CS 1 2 3 4 / US 5 6 7 8 / TR 9 10 11 12
    n = size(MM_10_order, 3);
    
    % trial randomization
    trial_rnd = randperm(n);
    
    % insert concatenated trials into 3rd element of 3D data
    MM_10 = MM_10_order(:, :, trial_rnd);
    
    save MM_10_1step

%% 2nd step: changing dimension from 3-d to 2-d in the data of each subject
% -----------------------------------------------------------------------------------
% 2-1) assigning variables for each trial
% 2-2) changing dimnsion of each trial to 2-d
% -----------------------------------------------------------------------------------

%% assigning variables for each trial
    t1 = MM_10(:, :, 1);
    t2 = MM_10(:, :, 2);
    t3 = MM_10(:, :, 3);
    t4 = MM_10(:, :, 4);
    t5 = MM_10(:, :, 5);
    t6 = MM_10(:, :, 6);
    t7 = MM_10(:, :, 7);
    t8 = MM_10(:, :, 8);
    t9 = MM_10(:, :, 9);
    t10 = MM_10(:, :, 10);
    t11 = MM_10(:, :, 11);
    t12 = MM_10(:, :, 12);

%% changing dimnsion of each trial to 2-d
 d2_MM_10 = cat(2, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12);
    
    save MM_10_2step 
end
