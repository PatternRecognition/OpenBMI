function [LOSS, CSP, LDA] = cal_loss( file )
%CAL_LOSS Summary of this function goes here
%   Detailed explanation goes here

marker={'1','left';'2','right';'3','foot'};
[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', 100});

field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};

LOSS=cell(3,2);
CSP=cell(3,2);
LDA=cell(3,2);
%% right vs left
CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
CNT=prep_selectClass(CNT,{'class',{'right', 'left'}});
%% PRE-PROCESSING MODULE
filter=[7 13];
CNT=prep_filter(CNT, {'frequency', filter});
SMT=prep_segmentation(CNT, {'interval', [1010 3000]});

%% SPATIAL-FREQUENCY OPTIMIZATION MODULE
[SMT, CSP_W, CSP_D]=func_csp(SMT,{'nPatterns', [3]});
FT=func_featureExtraction(SMT, {'feature','logvar'});

%% CLASSIFIER MODULE
[CF_PARAM]=func_train(FT,{'classifier','LDA'});


CV.prep={ % commoly applied to training and test data before data split
    'CNT=prep_filter(CNT, {"frequency", [7 13]})'
    'SMT=prep_segmentation(CNT, {"interval", [750 3500]})'
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
CSP{1,1}=CSP_W; CSP{1,2}='right vs left';
LDA{1,1}=CF_PARAM; LDA{1,2}='right vs left';
LOSS{1,1}=loss;LOSS{1,2}='right vs left';


%% right vs foot
CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
CNT=prep_selectClass(CNT,{'class',{'right', 'foot'}});
%% PRE-PROCESSING MODULE
filter=[7 13];
CNT=prep_filter(CNT, {'frequency', filter});
SMT=prep_segmentation(CNT, {'interval', [1010 3000]});

%% SPATIAL-FREQUENCY OPTIMIZATION MODULE
[SMT, CSP_W, CSP_D]=func_csp(SMT,{'nPatterns', [3]});
FT=func_featureExtraction(SMT, {'feature','logvar'});

%% CLASSIFIER MODULE
[CF_PARAM]=func_train(FT,{'classifier','LDA'});


CV.prep={ % commoly applied to training and test data before data split
    'CNT=prep_filter(CNT, {"frequency", [7 13]})'
    'SMT=prep_segmentation(CNT, {"interval", [750 3500]})'
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

CSP{2,1}=CSP_W; CSP{2,2}='right vs foot';
LDA{2,1}=CF_PARAM; LDA{2,2}='right vs foot';
LOSS{2,1}=loss; LOSS{2,2}='right vs foot';
%% left vs foot

CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
CNT=prep_selectClass(CNT,{'class',{'left', 'foot'}});
%% PRE-PROCESSING MODULE
filter=[7 13];
CNT=prep_filter(CNT, {'frequency', filter});
SMT=prep_segmentation(CNT, {'interval', [1010 3000]});

%% SPATIAL-FREQUENCY OPTIMIZATION MODULE
[SMT, CSP_W, CSP_D]=func_csp(SMT,{'nPatterns', [3]});
FT=func_featureExtraction(SMT, {'feature','logvar'});

%% CLASSIFIER MODULE
[CF_PARAM]=func_train(FT,{'classifier','LDA'});



CV.prep={ % commoly applied to training and test data before data split
    'CNT=prep_filter(CNT, {"frequency", [7 13]})'
    'SMT=prep_segmentation(CNT, {"interval", [750 3500]})'
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

CSP{3,1}=CSP_W; CSP{3,2}='left vs foot';
LDA{3,1}=CF_PARAM; LDA{3,2}='left vs foot';
LOSS{3,1}=loss;LOSS{3,2}='left vs foot';
end

