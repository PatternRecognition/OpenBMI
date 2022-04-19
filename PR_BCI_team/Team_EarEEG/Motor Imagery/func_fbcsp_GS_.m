function acc_MI_FBCSP = func_fbcsp_GS(CNT_tr,CNT_te)

% channel_index = [1:32];
band = [4 8;8 12;12 16;16 20;20 24;24 28;28 32;32 36;36 40];
% time_interval = [1000 3500];
CSPFilter=2;
NUMfeat=4;

channel_index = [8 9 10 11 13 14 15 18 19 20 21 33 34 35 36 37 38 39 40 41];
% band = [8 30];
time_interval = [1000 3500];

% Pre-processing
CNT_off = prep_selectChannels(CNT_tr, {'Index', channel_index});
CNT_off =prep_filterbank(CNT_off , {'frequency', band});
SMT_off = prep_segmentation_filterbank(CNT_off, {'interval', time_interval});

CNT_on= prep_selectChannels(CNT_te, {'Index', channel_index});
CNT_on =prep_filterbank(CNT_on , {'frequency', band});
SMT_on = prep_segmentation_filterbank(CNT_on, {'interval', time_interval});

% Feature extracion and Classification
[SMT, CSP_W, CSP_D]=func_csp_filterbank(SMT_off,{'nPatterns', CSPFilter});
FT=func_featureExtraction_filterbank(SMT, {'feature','logvar'});
[FT,idx]=func_MIBIF_filterbank(FT,{'nFeatures',NUMfeat});
[CF_PARAM]=func_train(FT,{'classifier','LDA'});

SMT_te=func_projection_filterbank(SMT_on, CSP_W);
FT_te=func_featureExtraction_filterbank(SMT_te, {'feature','logvar'});
FT_te=func_featureSelection_filterbank(FT_te,{'index',idx});
[cf_out]=func_predict(FT_te, CF_PARAM);

[loss out]=eval_calLoss(FT_te.y_dec, cf_out);
acc_MI_FBCSP=1-loss';
fprintf('FBCSP: %.2f%%\n', acc_MI_FBCSP);




end