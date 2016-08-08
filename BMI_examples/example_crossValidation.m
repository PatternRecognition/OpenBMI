clc
close all;
clear all;
OpenBMI('C:\Users\CVPR\Desktop\Open_Github') % Edit the variable BMI if necessary
global BMI;
BMI.EEG_DIR=['C:\Users\CVPR\Desktop\¿ø¾Ó\OpenBMI\data'];

%% DATA LOAD MODULE
file=fullfile(BMI.EEG_DIR, '\hblee_160627_calib_short');
marker={'1','left';'2','right';'3','foot'};
[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', 500});

field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
CNT=prep_selectClass(CNT,{'class',{'right', 'left'}});


%% CROSS-VALIDATION MODULE
CV.var.band=[10 13];
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
'KFold','5'
% 'leaveout'
};

[loss]=eval_crossValidation(CNT, CV); % input : eeg, or eeg_epo