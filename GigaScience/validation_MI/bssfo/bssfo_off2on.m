function Acc = bssfo_off2on(CNT,general_initparam,bssfo_param)
opt = opt_cellToStruct(general_initparam);
bssfo_opt = opt_cellToStruct(bssfo_param);

% Pre-processing
%Train
CNT_off = prep_selectChannels(CNT{1}, {'Index', opt.channel_index});
CNT_off =prep_filter(CNT_off , {'frequency', bssfo_opt.init_band});
SMT_off = prep_segmentation(CNT_off, {'interval', opt.time_interval});

%Test
CNT_on= prep_selectChannels(CNT{2}, {'Index', opt.channel_index});
CNT_on =prep_filter(CNT_on , {'frequency', bssfo_opt.init_band});
SMT_on = prep_segmentation(CNT_on, {'interval', opt.time_interval});
%% BSSFO - optimal band and weight
[FilterBand]=func_bssfo(SMT_off, {'classes', {'right', 'left'}; ...
    'frequency', {bssfo_opt.mu_band,bssfo_opt.beta_band}; 'std', {5, 25}; 'numBands', bssfo_opt.numBands; ...
    'numCSPPatterns', opt.CSPFilter; 'numIteration', bssfo_opt.numIteration});

%% BSSFO - CSP filter from optimal band
for iii=1:bssfo_opt.numIteration
    CNTtr=prep_filter(CNT{1}, {'frequency', FilterBand.sample(:,iii)'});
    CNTtr= prep_selectChannels(CNTtr, {'Index', opt.channel_index});
    SMT=prep_segmentation(CNTtr, {'interval', opt.time_interval});
    [SMT, CSP_W, CSP_D]=func_csp(SMT,{'nPatterns', opt.CSPFilter});
    FT=func_featureExtraction(SMT, {'feature','logvar'});
    [CF_PARAM]=func_train(FT,{'classifier','LDA'});
    clear CNTclass CNTtr SMT FT CSP_D
    
    CNTte=prep_filter(CNT{2}, {'frequency', FilterBand.sample(:,iii)'});
    CNTte= prep_selectChannels(CNTte, {'Index', opt.channel_index});
    SMTte=prep_segmentation(CNTte, {'interval', opt.time_interval});
    SMTfb=func_projection(SMTte, CSP_W);
    FTfb=func_featureExtraction(SMTfb, {'feature','logvar'});
    [cf_out]=func_predict(FTfb, CF_PARAM);
    a(:,iii)=cf_out.*FilterBand.weight(iii);
    clear CNTte SMTte FTfb CSP_W
end
A = sum(a,2);
%%
[loss out]=eval_calLoss(CNT{2}.y_dec, A);
Acc=1-loss;
end