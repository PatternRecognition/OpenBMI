clear all; close all; clc;
%% Initialization 
%data load
% name={'1_bykim','2_dblee','3_eskim','9_prchoi','10_smkang','12_yskim'}; %session 0
name={'1_bykim','2_dblee','3_eskim','4_jhsuh','5_mgjung','6_mhlee','7_mjkim','8_oykwon','9_prchoi','10_smkang','11_spseo','12_yskim','13_jyha','14_hmyoo','15_jwlee','16_wbsim','17_hwnoh','18_dysull','19_jyhan','20_jwbae','21_syjung','22_dkhan','23_hjwon','24_nkkil','25_khshim'}; % session 1
% name={'1_bykim','2_dblee','3_eskim','4_jhsuh','5_mgjung','7_mjkim','9_prchoi','10_smkang','12_yskim','13_jyha','14_hmyoo','15_jwlee','16_wbsim','17_hwnoh','18_dysull','19_jyhan','20_jwbae','21_syjung','22_dkhan','23_hjwon'}; % session 2
session = {'session1'};
task = {'mi_off','mi_on'};
fs=100; 

%pre-processing
channel_index = [1:32];
band = [8 30];
time_interval = [1000 3500];

%feature-extraction
CSPFilter=2;
%classifiaction

%% Data load and mat save
for sub=1:length(name)
    for onoff=1:2
        file3 = ['E:\Users\cvpr\Desktop\StarlabDB_2nd\',name{sub},'\'];
        BMI.EEG_DIR=['E:\Users\cvpr\Desktop\StarlabDB_2nd\',name(sub),'\',session];
        BMI.EEG_DIR=cell2mat(BMI.EEG_DIR);
        file=fullfile(BMI.EEG_DIR, task{onoff});
        marker={'1','left';'2','right'};
        [EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', fs});
        field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
        CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
        CNTT{sub,onoff} = CNT;
    end
    %     filename1= ['mi_cnt_s1_on'];
    %     save([file3, filename1], 'CNT');
end
%% 'mat' load
% for sub=1:11
%     
% %     name={'bykim','dblee','ejlee','eskim','mhlee','prchoi','sbsim','smkang','yelee','yskim'};
%     name={'bykim','eskim','prchoi','smkang','oykwon'};
%     file3 = ['E:\Users\cvpr\Desktop\StarlabDB_2nd\',name{sub},'\'];    
%     filename4 = ['mi_cnt_s1_on.mat'];       
%     
%     cnt = load([file3, filename4]);   
%     % MI 19 ch
%     cnt= prep_selectChannels(cnt.CNT, {'Name',{'FC5','FC3','FC1','C5','C3','C1','Cz','CP5','CP3','CP1','CP6','CP4','CP2','C6','C4','C2','FC6','FC4','FC2'}});
%     cnt=prep_filter(cnt, {'frequency', [7 13]});
%     new_smt{sub,1}=prep_segmentation(cnt, {'interval', [750 3500]});
%     
% end
%% Pre-processing
clear CNT clear BMI ans EEG field file file3 marker sub
for NUM=1:length(CNTT)
    for onoff=1:2
        CNTch = prep_selectChannels(CNTT{NUM,onoff}, {'Index', channel_index});
        CNTchfilt =prep_filter(CNTch , {'frequency', band});
        all_SMT{NUM,onoff} = prep_segmentation(CNTchfilt, {'interval', time_interval});
        clear CNTch CNTchfilt
    end
end
%% Feature extraction and classification
%% CSP - LDA
for num = 1:length(all_SMT)
    [SMT, CSP_W, CSP_D]=func_csp(all_SMT{num,1},{'nPatterns', CSPFilter});
    FT=func_featureExtraction(SMT, {'feature','logvar'});
    [CF_PARAM]=func_train(FT,{'classifier','LDA'});    
    SMT_te=func_projection(all_SMT{num,2}, CSP_W);
    
    FT_te=func_featureExtraction(SMT_te, {'feature','logvar'});
    [cf_out]=func_predict(FT_te, CF_PARAM);
    [loss out]=eval_calLoss(FT_te.y_dec, cf_out);
    sub_acc(:,num)=1-loss';    
    clear SMT CSP_W CSP_D FT CF_PARAM SMT_te FT_Te cf_out loss out 
end
    all_acc =sub_acc';
    all_mean_acc = mean(sub_acc,2);
%% FBCSP - LDA

%% BSSFO - LDA





