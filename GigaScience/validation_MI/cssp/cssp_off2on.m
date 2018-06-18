function [ Acc ] = cssp_off2on(CNT,general_initparam,tau)

opt = opt_cellToStruct(general_initparam);
% Pre-processing
train_raw = prep_selectChannels(CNT{1}, {'Index', opt.channel_index});
train_cnt =prep_filter(train_raw  , {'frequency', opt.band});
train_smt = prep_segmentation(train_cnt, {'interval', opt.time_interval});

test_raw= prep_selectChannels(CNT{2}, {'Index', opt.channel_index});
test_cnt =prep_filter(test_raw , {'frequency', opt.band});
test_smt = prep_segmentation(test_cnt, {'interval', opt.time_interval});

% Feature extracion and Classification
for len_tau=1:length(tau)
    CV.train={
        '[SMT, CSP_W, CSP_D]=func_csp(SMT,{"nPatterns", [2]})'
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
        };
    SMT=func_TDE(train_smt,train_cnt,tau(len_tau),opt.time_interval);
    [loss]=eval_crossValidation(SMT, CV); % input : eeg, or eeg_epo
    tau_result(1,len_tau)=1-loss;
end
[acc_cv,max_tau]=max(tau_result);
ttau=tau(max_tau);

SMT_tr=func_TDE(train_smt,train_cnt,ttau,opt.time_interval);
[SMT_tr, CSP_W, CSP_D]=func_csp(SMT_tr,{'nPatterns', opt.CSPFilter});
FT_tr=func_featureExtraction(SMT_tr, {'feature','logvar'});
[CF_PARAM]=func_train(FT_tr,{'classifier','LDA'});

SMT_te=func_TDE(test_smt,test_cnt,ttau,opt.time_interval);
SMT_te=func_projection(SMT_te, CSP_W);
FT_te=func_featureExtraction(SMT_te, {'feature','logvar'});
[cf_out]=func_predict(FT_te, CF_PARAM);

[loss out]=eval_calLoss(FT_te.y_dec, cf_out);
Acc=1-loss';
% fprintf('CSSP: %.2f%%\n', Acc);


end