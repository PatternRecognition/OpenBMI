% Motor imagery - 10-fold-cross-validation with 10-iterations

load cnt_mi;

% cnt variables
% cnt.t  : time information 
% cnt.fs : sampling frequency
% cnt.y_dec : class information (e.g., left = 1, right = 2)
% cnt.y_logic : logical format of class inforamtion 
% cnt_y_class : class name (e.g., left, right)
% cnt.class : number of class 
% cnt.chan : number of electrodes
% cnt. x : raw eeg signals

% revised 2017.11.11 - Oyeon Kwon (oy_kwon@korea.ac.kr)
%% Initialization
freq_band = [7 13];
interval = [750 3500];
%% Analysis
CNT = cnt_mi;
CNTfilt =prep_filter(CNT , {'frequency', freq_band});
SMT = prep_segmentation(CNTfilt, {'interval', interval});

for iter=1:10
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
'KFold','10'
% 'leaveout'
};

[loss]=eval_crossValidation(SMT, CV); % input : eeg, or eeg_epo
Result_iter(1,iter)=1-loss';
end

Mean_CV_result=mean(Result_iter',1)';
