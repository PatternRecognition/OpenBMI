clear all; clc; close all;

OpenBMI('G:\OpenBMI') % Edit the variable BMI if necessary
global BMI;
BMI.EEG_DIR=['G:\OpenBMI\BMI_data\RawEEG'];

%% DATA LOAD MODULE for competetion_4a (2-class)
filename={'data_set_IVa_aa.mat','data_set_IVa_al.mat','data_set_IVa_av.mat','data_set_IVa_aw.mat','data_set_IVa_ay.mat'}
true={'true_labels_aa','true_labels_al','true_labels_av','true_labels_aw','true_labels_ay'}
Train=[168, 224, 84, 56, 28];
i=5;
% for i=1:5
Competetion=load(filename{i});
load(true{i});
Competetion.mrk.y=true_y;

Converting.x=double(Competetion.cnt);
Converting.t=Competetion.mrk.pos;
Converting.fs=Competetion.nfo.fs;

Converting.y_dec=zeros(size(Competetion.mrk.y));
idx1=Competetion.mrk.y==1;
idx2=Competetion.mrk.y==2;
Converting.y_dec(1,idx1)=1;
Converting.y_dec(2,idx2)=1;
Converting.y_logic=logical(Converting.y_dec);
Converting.y_dec=true_y;
Converting.y_class=cell(size(true_y));

lable_1=find(true_y==1);
Converting.y_class(lable_1)=Competetion.mrk.className(1);
lable_2=find(true_y==2);
% strcmp(CNT.y_class,'right');
Converting.y_class(lable_2)=Competetion.mrk.className(2);

Converting.class={1,Competetion.mrk.className{1,1}; 2,Competetion.mrk.className{1,2}};
Converting.class={1,Competetion.mrk.className{1,1}; 2,Competetion.mrk.className{1,2}}
Converting.chan=Competetion.nfo.clab;

%% filtering
Converting=prep_filter(Converting, {'frequency', [7 13]});
Converting=prep_segmentation(Converting, {'interval', [750 3500]});
prep_selectClass(Converting,{'class',{'right','foot'}})

%% test
train=prep_selectTrials(Converting,{'index',[1:Train(i)]})
test=prep_selectTrials(Converting,{'index',[Train(i)+1:280]})

[SMT, CSP_W, CSP_D]=func_csp(train,{'nPatterns',[3]});
FT=func_featureExtraction(SMT, {'feature','logvar'});
[CF_PARAM]=func_train(FT,{'classifier','LDA'});

SMTfb=func_projection(test, CSP_W);
FTfb=func_featureExtraction(SMTfb, {'feature','logvar'});
[cf_out]=func_predict(FTfb, CF_PARAM);
[loss out]=eval_calLoss(FTfb.y_dec, cf_out);
