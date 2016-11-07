clear all;close all;clc
% restoredefaultpath

% p=genpath('C:\Users\CVPR\Desktop\CVPR\StarLab\GitHub\OpenBMI');
% addpath(p)
% addpath(genpath('C:\Users\CVPR\Desktop\CVPR\StarLab\smkim_functions'))

dir = 'E:\Data\Data_backup\[2013]EEG-NIRS\EEG_RAW';
sub={'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n'};
date={'_10_01_13','_10_01_15','_10_01_25','_10_01_27','_10_03_05', ...
    '_10_03_24','_10_06_17','_10_06_19','_10_06_18','_10_06_22', ...
    '_10_07_19','_10_07_20','_10_07_21','_10_07_26'};

%% Values setting
% Learning rate
UC = 0.05;
% Frequency band for filtering
% freq=[10.5 14.5];
% Time interval for segmentingacc_basic
time=[0 4000];
% # training/test
n_tr=100;n_te=100;

acc_basic=zeros(1,length(sub));
acc_pmean=zeros(1,length(sub));
acc_pmean30=zeros(1,length(sub));

for s=2:length(sub)
%% Load data
DIR=[dir,'\VPea' sub{s},date{s},'\imag_fbarrow_pcovmeanVPea',sub{s}];
file=fullfile(DIR);

marker={'1','left';'2','right'};
[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker',marker;'fs',100});
field={'x','t','fs','orig_fs','y_dec','y_logic','y_class','class','chan'};
% field={'x','t','fs','y_dec','y_logic','y_class','class','chan'};
cnt1=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);

DIR2=[DIR,'02'];
file=fullfile(DIR2);
[EEG2.data, EEG2.marker, EEG2.info]=Load_EEG(file,{'device','brainVision';'marker',marker;'fs',100});
cnt2=opt_eegStruct({EEG2.data, EEG2.marker, EEG2.info}, field);

%% Pre-processing
% freq=[8 12]
% freq=[4 8]
% % cnt1=prep_filter(cnt1, {'frequency', freq});
cnt1=prep_selectClass(cnt1,{'class',{'left','right'}});
smt1=prep_segmentation(cnt1, {'interval', time});

[FilterBand]=func_bssfo(smt1, {'classes', {'right', 'left'};'frequency', {[7 13],[14 30]}; 'std', {5, 25}; ...
    'numBands', 30; 'numCSPPatterns', 2; 'numIteration', 30});
freq=FilterBand.sample(:,1);


flt_cnt1=prep_filter(cnt1, {'frequency', freq});
smt1=prep_segmentation(flt_cnt1, {'interval', time});


flt_cnt2=prep_filter(cnt2, {'frequency', freq});
flt_cnt2=prep_selectClass(flt_cnt2,{'class',{'left','right'}});
smt2=prep_segmentation(flt_cnt2, {'interval', time});

%% CSP
[fv_train, csp_w, csp_d]=func_csp(smt1,{'nPatterns', 3});
fv_train=func_featureExtraction(fv_train, {'feature','logvar'});

%% Train LDA
[cls_param]=func_train(fv_train,{'classifier','LDA'});

%% Feature extraction for test data
fv_test = func_projection(smt2,csp_w);
fv_test = func_featureExtraction(fv_test,{'feature','logvar'});

%% Classification
% basic
cls_out=func_predict(fv_test, cls_param);
[loss,out]=eval_calLoss(fv_test.y_dec, cls_out);

% % pmean (after each trial)
% label_pmean=zeros(1,n_te);
% m=(mean(fv_train.x(:,smt1.y_dec==1),2)+mean(fv_train.x(:,smt1.y_dec==2),2))/2;
% param=cls_param;
% for i=1:n_te % adapt after each trial
%     [param,m,label_pmean(i)] = adaptLDA(smt2.x(:,i,:),csp_w,param,m,{'UC',UC});
% end

% % pmean (after every 30 trial)
% w30=zeros(size(csp_w,2),ceil(n_te/30)); % each trial
% b30=zeros(1,ceil(n_te/30));
% label_pmean30=cell(1,ceil(n_te/30));
% m30=(mean(fv_train.x(:,smt1.y_dec==1),2)+mean(fv_train.x(:,smt1.y_dec==2),2))/2;
% param30=cls_param;
% for i=1:n_te/30 % adaptLDA input에는 adapt하는 한 세트씩
%     [up_param30,m30,label_pmean30{i}] = adaptLDA(smt2.x(:,((30*(i-1)+1):30*i),:),csp_w,param30,m30,{'UC',UC;'Nadapt',30});
%     w30(:,i)=up_param30.w;param30.cf_param.w30=up_param30.w;
%     b30(i)=up_param30.b;param30.cf_param.b30=up_param30.b;
% end
% label_pmean30=cell2mat(label_pmean30);

%% Accuracy
acc_basic(s)=1-loss
% acc_pmean(s)=sum(label_pmean==smt2.y_dec)/n_te;
% acc_pmean30(s)=sum(label_pmean30==smt2.y_dec(1:floor(n_te/30)*30))/n_te;
end


% [FilterBand]=func_bssfo(SMT, {'classes', {'right', 'left'};'frequency', {[7 15],[14 30]}; 'std', {5, 25}; ...
%     'numBands', 30; 'numCSPPatterns', 2; 'numIteration', 30});
