clear all; close all; clc;
%% Initialization 
%data load
% name={'1_bykim','2_dblee','3_eskim','9_prchoi','10_smkang','12_yskim'}; %session 0
% name={'1_bykim','2_dblee','3_eskim','4_jhsuh','5_mgjung','6_mhlee','7_mjkim','8_oykwon','9_prchoi','10_smkang','11_spseo','12_yskim','13_jyha','14_hmyoo','15_jwlee','16_wbsim','17_hwnoh','18_dysull','19_jyhan','20_jwbae','21_syjung','22_dkhan','23_hjwon','24_nkkil','25_khshim'}; % session 1
% name={'1_bykim','2_dblee','3_eskim','4_jhsuh','5_mgjung','7_mjkim','9_prchoi','10_smkang','12_yskim','13_jyha','14_hmyoo','15_jwlee','16_wbsim','17_hwnoh','18_dysull','19_jyhan','20_jwbae','21_syjung','22_dkhan','23_hjwon'}; % session 2
name={'2_dblee'};
session = {'session1'};
task = {'mi_off','mi_on'};
fs=100; 

%pre-processing
channel_index = [1:32];
band = [4 40];
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
        marker={'1','left';'2','right';'78','rest'};
        [EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', fs});
        field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
        CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
        CNTT{sub,onoff} = CNT;
    end
    %     filename1= ['mi_cnt_s1_on'];
    %     save([file3, filename1], 'CNT');
end
%%
for i=1
CNTclass = prep_selectClass(CNTT{1,i},{'class',{'left', 'right'}});
CNT=prep_filter(CNTclass, {'frequency', [4 40]});
CNT= prep_selectChannels(CNT, {'Index', channel_index});
SMT=prep_segmentation(CNT, {'interval', time_interval});
end


%% BSSFO - optimal band and weight
[FilterBand]=func_bssfo(SMT, {'classes', {'left', 'right'}; ...
    'frequency', {[7 15],[14 30]}; 'std', {5, 25}; 'numBands', 50; ...
    'numCSPPatterns', 2; 'numIteration', 10});
clear SMT CNT CNTclass

select_BSSFO = band_check(FilterBand);
%% BSSFO - CSP filter from optimal band
% weight * LDA score
% Output: n개의 particle : n개의 optimal band 그리고 n개의 weight
% 과정 1. n개의 밴드로부터 CSP filter 구하기
% 과정 2. 각 트라이얼마다 이 CSP filter 곱해서 log-var 씌우고 --> feature vector 구하기
% 과정 3. 구한 feature vector에 SVM score 구하고 * weight 곱해주기
% 과정 4. 과정 3에서 곱해진 값들을 선택된 개수만큼 모두 더한뒤 argmax
% 과정 5. argmax한뒤 예측한 클래스가 0인지 1인지 판단. 이것을 바탕으로 정확도 계산

% opt_band = FilterBand.sample';
opt_band = select_BSSFO.sample';
for iii=1:length(opt_band);
    CNTclass = prep_selectClass(CNTT{1,1},{'class',{'left', 'right'}});
    CNT=prep_filter(CNTclass, {'frequency', opt_band(iii,:)});
    CNT= prep_selectChannels(CNT, {'Index', channel_index});
    % Laplace
    SMT=prep_segmentation(CNT, {'interval', time_interval});
    % baseline
    [SMT, CSP_W, CSP_D]=func_csp(SMT,{'nPatterns', [2]});
    FT=func_featureExtraction(SMT, {'feature','logvar'});
    [CF_PARAM]=func_train(FT,{'classifier','LDA'}); % need to change SVM
    % FT=func_featureExtraction(SMT, {'feature','logvar'});
    % X_TR = FT.x';
    % group = FT.y_class';
    % svmStruct = svmtrain(X_TR,group,'kernel_function','rbf','ShowPlot',true);
    %% Projecting test data to CSP filter
    CNTclasste = prep_selectClass(CNTT{1,2},{'class',{'left', 'right'}});
    CNTte=prep_filter(CNTclasste, {'frequency', opt_band(iii,:)});
    CNTte= prep_selectChannels(CNTte, {'Index', channel_index});
    % Laplace
    SMTte=prep_segmentation(CNTte, {'interval',  time_interval});
    SMTfb=func_projection(SMTte, CSP_W);
    FTfb=func_featureExtraction(SMTfb, {'feature','logvar'});
    [cf_out]=func_predict(FTfb, CF_PARAM);
    
%         a(:,iii)=cf_out.*FilterBand.weight(iii); %BSSFO accuracy
    a(:,iii)=cf_out.*select_BSSFO.weight(iii); %BSSFO accuracy
    clear CNTclass CNT SMT CSP_W CSP_D FT CF_PARAM CNTclasste CNTte SMTte SMTfb FTfb cf_out
end
A = sum(a,2);
%%
CNTclasste = prep_selectClass(CNTT{1,2},{'class',{'left', 'right'}});
[loss out]=eval_calLoss(CNTclasste.y_dec, A);
1-loss
clear CNTclasste 



