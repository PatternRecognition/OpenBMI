function [ Acc ] = mi_performance_off2on(CNT,general_initparam)
%CNT{1} = train
%CNT{2} = test
opt = opt_cellToStruct(general_initparam);

% Pre-processing
for onoff=1:length(opt.task)
    CNTch = prep_selectChannels(CNT{onoff}, {'Index', opt.channel_index});
    CNTchfilt =prep_filter(CNTch , {'frequency', opt.band});
    all_SMT{onoff} = prep_segmentation(CNTchfilt, {'interval', opt.time_interval});
    clear CNTch CNTchfilt
end
train = all_SMT{1}; test = all_SMT{2};

% Feature extracion and Classification
[SMT, CSP_W, CSP_D]=func_csp(train,{'nPatterns', opt.CSPFilter});
FT=func_featureExtraction(SMT, {'feature','logvar'});
[CF_PARAM]=func_train(FT,{'classifier','LDA'});

SMT_te=func_projection(test, CSP_W);
FT_te=func_featureExtraction(SMT_te, {'feature','logvar'});
[cf_out]=func_predict(FT_te, CF_PARAM);
[loss out]=eval_calLoss(FT_te.y_dec, cf_out);
Acc=1-loss;
end

