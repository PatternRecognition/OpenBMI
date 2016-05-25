clear all;%close all;clc
restoredefaultpath

p=genpath('C:\Users\CVPR\Desktop\CVPR\StarLab\GitHub\OpenBMI');
addpath(p)
addpath(genpath('C:\Users\CVPR\Desktop\CVPR\StarLab\smkim_functions'))

dir = 'C:\Users\CVPR\Desktop\CVPR\StarLab\Data\bbciRaw\VPea';
sub={'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n'};
date={'_10_01_13','_10_01_15','_10_01_25','_10_01_27','_10_03_05', ...
    '_10_03_24','_10_06_17','_10_06_19','_10_06_18','_10_06_22', ...
    '_10_07_19','_10_07_20','_10_07_21','_10_07_26'};

%% Values setting
% Learning rate
UC = 0.05;
% # of training data
n_tr = 50;
% Frequency band for filtering
freq=[10.5 14.5];
% Time interval for segmenting
time=[-1000 4000];
% adaptation interval (# trials)
% adapt_n=1;

% for i=1:length(sub)
i=14;
%% Load data
DIR=[dir,sub{i},date{i},'\imag_fbarrow_pcovmeanVPea',sub{i}];
file=fullfile(DIR);
marker={'1','left';'2','right'};
[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker',marker;'fs',100});
field={'x','t','fs','orig_fs','y_dec','y_logic','y_class','class','chan'};
cnt1=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);

DIR2=[DIR,'02'];
file=fullfile(DIR2);
[EEG2.data, EEG2.marker, EEG2.info]=Load_EEG(file,{'device','brainVision';'marker',marker;'fs',100});
cnt2=opt_eegStruct({EEG2.data, EEG2.marker, EEG2.info}, field);

cnt = prep_addTrials(cnt1,cnt2);
cnt.t(101:end)=cnt.t(101:end)+cnt.t(100);

%% Pre-processing
cnt=prep_filter(cnt, {'frequency', freq});
cnt=prep_selectClasses(cnt,{'Class',{'left','right'}});
smt=prep_segmentation(cnt, {'interval', time});
% # of test data
n_te = size(smt.x,2)-n_tr;

%% Dividing the data into training & test set
train = prep_selectTrials(smt,{'Index',1:n_tr});
test = prep_selectTrials(smt,{'Index',n_tr+1:size(smt.x,2)});

%% CSP
% [fv_train,csp_w] = func_csp(train);
% fv_train.x = squeeze(log(var(fv_train.x)));
[fv_train, csp_w, csp_d]=func_csp(train,{'nPatterns', 3});
fv_train=func_featureExtraction(fv_train, {'feature','logvar'});

%% Train LDA
% param = train_RLDAshrink(fv_train.x',fv_train.y_logic); % bbci
[cls_param]=func_train(fv_train,{'classifier','LDA'});
% param=cls_param.cf_param;

%% Feature extraction for test data
% [nt,ntr,nch]=size(test.x);
% fv_test=squeeze(log(var(reshape(reshape(test.x,[nt*ntr,nch])*csp_w,[nt,ntr,size(csp_w,2)]))));
fv_test = func_projection(test,csp_w);
fv_test = func_featureExtraction(fv_test,{'feature','logvar'});

%% Basic classification
% label_basic=real(param.w'*fv_test' + param.b*ones(size(1,n_te)));
% for i=1:size(label_basic,2)
%     if label_basic(i)<0,label_basic(i)=1;
%     else label_basic(i)=2;
%     end
% end
cls_out=func_predict(fv_test, cls_param);
[loss,out]=eval_calLoss(fv_test.y_dec, cls_out);

%% Apply adaptation (pmean) for test data
w=zeros(size(csp_w,2),n_te); % each trial
b=zeros(1,n_te);
label_pmean=zeros(1,n_te);
m=(mean(fv_train.x(:,train.y_dec==1),2)+mean(fv_train.x(:,train.y_dec==2),2))/2;
param=cls_param.cf_param;
for i=1:n_te % adapt after each trial
    [up_param,m,label_pmean(i)] = adaptLDA(test.x(:,i,:),csp_w,param,m,{'UC',UC});
    w(:,i)=up_param.w;param.w=up_param.w;
    b(i)=up_param.b;param.b=up_param.b;
end

% % +++ every 30 trial
% w30=zeros(size(csp_w,2),ceil(n_te/30)); % each trial
% b30=zeros(1,ceil(n_te/30));
% label_pmean30=zeros(1,ceil(n_te/30));
% m30=(mean(fv_train.x(:,train.y_dec==1),2)+mean(fv_train.x(:,train.y_dec==2),2))/2;
% param30=cls_param.cf_param;
% for i=1:n_te/30 % adaptLDA input에는 adapt하는 한 세트씩
%     [up_param30,m30,label_pmean30(i)] = adaptLDA(test.x(:,((30*(i-1)+1):30*i),:),csp_w,param30,m30,{'UC',UC;'Nadapt',30});
%     w30(:,i)=up_param30.w;param30.w30=up_param30.w;
%     b30(i)=up_param30.b;param30.b30=up_param30.b;
% end

%% Apply BSSFO for test data
opt_freq=[];


%% Accuracy (percentage of correct accomplished trials)
% acc_basic=sum(label_basic==test.y_dec)/n_te;
acc_basic=1-loss;
acc_pmean=sum(label_pmean==test.y_dec)/n_te;

% end