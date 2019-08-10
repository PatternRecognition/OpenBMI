clear all; clc;
%% load matfile 
% load('C:\Users\Oyeon Kwon\Documents\MATLAB\Deeplearning\bbci_IV_2a_mat\CNTT_off.mat');
% load('C:\Users\Oyeon Kwon\Documents\MATLAB\Deeplearning\bbci_IV_2a_mat\CNTT_on.mat');

CNTT= CNTT;
channel_index = [1:32];
% channel_index = [1:56 61:66];
time_interval = [1000 3500];
band = [4 8;8 12;12 16;16 20;20 24;24 28;28 32;32 36;36 40];
CSPFilter = 2;
NUMfeat=4;
%% main
% Pre-processing
for sub=1:length(CNTT)
    CNT_off=CNTT{sub,1}; % offline  
    CNT_on=CNTT{sub,2};  % 
    
    CNT_off = prep_selectChannels(CNT_off, {'Index', channel_index});
    CNT_off = prep_selectClass(CNT_off,{'class',{'right', 'left'}});
    CNT_off =prep_filterbank(CNT_off , {'frequency', band});
    SMT_off = prep_segmentation_filterbank(CNT_off, {'interval', time_interval});
    
    CNT_on= prep_selectChannels(CNT_on, {'Index', channel_index});
    CNT_on = prep_selectClass(CNT_on,{'class',{'right', 'left'}});
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
 
% if use other class information 
%     FT_te.y_dec;
%     label1=str2num(FT_te.class{1,1});
%     [a b] =find(FT_te.y_dec(:) == label1);
%     FT_te.y_dec(a) =1;
%     
%     label2=str2num(FT_te.class{2,1});
%     [aa bb ] = find(FT_te.y_dec(:) == label2);
%     FT_te.y_dec(aa) =2;
    
    [loss out]=eval_calLoss(FT_te.y_dec, cf_out);
    acc(sub)=1-loss;
end

mean(acc);
