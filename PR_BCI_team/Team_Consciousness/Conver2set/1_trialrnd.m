% 1step: trial randomization in the same subject
% -----------------------------------------------------------------------------------
% 1-1) struct -> arracy
% 1-2) modifying the length of epoch: 10sec for each class -> 5 sec

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
    
    %% UCS epoch 1과 2 합치기
    epo_MM_UCS = cat(3, epo1_MM_UCS, epo2_MM_UCS);
    
    %% WFN random 추출
    % WFN trial을 random으로 2개 뽑고, 하나로 합치기
    MM_WFN_1 = epo_MM_WFN(:, :, randperm(29, 1));
    MM_WFN_2 = epo_MM_WFN(:, :, randperm(29, 1));

    MM_WFN_rnd = cat(3, MM_WFN_1, MM_WFN_2);    
    
    %% LOC random 추출
    % LOC trial을 random으로 2개 뽑고, 하나로 합치기
    MM_LOC_1 = epo_MM_LOC(:, :, randperm(3, 1));
    MM_LOC_2 = epo_MM_LOC(:, :, randperm(3, 1));
    
    MM_LOC_rnd = cat(3, MM_LOC_1, MM_LOC_2);  
    
    %% UCS random 추출
    % UCS trial을 random으로 4개 뽑고, 하나로 합치기 
    MM_UCS_1 = epo_MM_UCS(:, :, randperm(6, 1));
    MM_UCS_2 = epo_MM_UCS(:, :, randperm(6, 1));
    MM_UCS_3 = epo_MM_UCS(:, :, randperm(6, 1));
    MM_UCS_4 = epo_MM_UCS(:, :, randperm(6, 1));
   
    MM_UCS_rnd = cat(3, MM_UCS_1, MM_UCS_2, MM_UCS_3, MM_UCS_4);
    
    %% ROC random 추출
    % ROC trial을 random으로 2개 뽑고, 하나로 합치기
    MM_ROC_1 = epo_MM_ROC(:, :, randperm(3, 1));
    MM_ROC_2 = epo_MM_ROC(:, :, randperm(3, 1));
   
    MM_ROC_rnd = cat(3, MM_ROC_1, MM_ROC_2);  
    
    %% REV random 추출
    % REV trial을 random으로 2개 뽑고, 하나로 합치기
    MM_REV_1 = epo_MM_REV(:, :, randperm(30, 1));
    MM_REV_2 = epo_MM_REV(:, :, randperm(30, 1));

    MM_REV_rnd = cat(3, MM_REV_1, MM_REV_2);
