% 1step: trial randomization in the same subject
% -----------------------------------------------------------------------------------
% 1-1) struct -> arracy
% 1-2) This step will be uploaded later..........


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
    
  
