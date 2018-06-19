function Acc = fbcsp_off2on(CNT,general_initparam,filterbank,NUMfeat)
opt = opt_cellToStruct(general_initparam);

% Pre-processing
CNT_off = prep_selectChannels(CNT{1}, {'Index', opt.channel_index});
CNT_off =prep_filterbank(CNT_off , {'frequency', filterbank});
SMT_off = prep_segmentation_filterbank(CNT_off, {'interval', opt.time_interval});

CNT_on= prep_selectChannels(CNT{2}, {'Index', opt.channel_index});
CNT_on =prep_filterbank(CNT_on , {'frequency', filterbank});
SMT_on = prep_segmentation_filterbank(CNT_on, {'interval', opt.time_interval});

% Feature extracion and Classification
[SMT, CSP_W, CSP_D]=func_csp_filterbank(SMT_off,{'nPatterns', opt.CSPFilter});
FT=func_featureExtraction_filterbank(SMT, {'feature','logvar'});
[FT,idx]=func_MIBIF_filterbank(FT,{'nFeatures',NUMfeat});
[CF_PARAM]=func_train(FT,{'classifier','LDA'});

SMT_te=func_projection_filterbank(SMT_on, CSP_W);
FT_te=func_featureExtraction_filterbank(SMT_te, {'feature','logvar'});
FT_te=func_featureSelection_filterbank(FT_te,{'index',idx});
[cf_out]=func_predict(FT_te, CF_PARAM);

[loss out]=eval_calLoss(FT_te.y_dec, cf_out);
Acc=1-loss';
end