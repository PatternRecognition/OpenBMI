clear all; clc; close all;

OpenBMI('C:\Users\CVPR\Desktop\Open_Github') % Edit the variable BMI if necessary
global BMI;
BMI.EEG_DIR=['G:\data2'];

%% DATA LOAD MODULE
file=fullfile(BMI.EEG_DIR, '\2016_08_05_hkkim_training');
marker= {'1','right';'2','left';'3','foot';'4','rest'};
[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', 500});

field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
CNT=prep_selectClass(CNT,{'class',{'right', 'rest'}});

CNT2 = prep_laplacian(CNT, {'Channel', {'C3', 'Cz', 'C4'}})
 
%% CROSS-VALIDATION MODULE
CV.var.band=[7 20];
CV.var.interval=[750 3500];
CV.prep={ % commoly applied to training and test data before data split
    'CNT=prep_filter(CNT, {"frequency", band})'
    'SMT=prep_segmentation(CNT, {"interval", interval})'
    };
CV.train={
    '[SMT, CSP_W, CSP_D]=func_csp(SMT,{"nPatterns", [3]})'
    'FT=func_featureExtraction(SMT, {"feature","logvar"})'
    '[CF_PARAM]=func_train(FT,{"classifier","LDA"})'
    };
CV.test={
    'SMT=func_projection(SMT, CSP_W)'
    'FT=func_featureExtraction(SMT, {"feature","logvar"})'
    '[cf_out]=func_predict(FT, CF_PARAM)'
    };
CV.option={
'KFold','7'
% 'leaveout'
};

[loss]=eval_crossValidation(CNT, CV); % input : eeg, or eeg_epo
