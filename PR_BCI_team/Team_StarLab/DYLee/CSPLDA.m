%% 데이터 입력형태 바꾸기
data_processing;

EEG_3itrans=permute(EEG_3i,[2 1 3]);
ds = downsample(EEG_3itrans,8);
EEG_3ids=permute(ds,[2 1 3 ]);
%% channel 정보
channel={'F3', 'F4', 'C3', 'C4', 'P3', 'P4'};

%% class mat 만들기
class=getclassvec_11(class_i');
%% epo data
epo.x=EEG_3ids;
% epo.x=EEG_3i;
% epo.x=EEG;
epo.y=class;
epo.clab=channel;

%% CSP - FEATURE EXTRACTION
%     [csp_fv,csp_w,csp_eig]=proc_multicsp(epo,3);
[csp_fv,csp_w,csp_eig]=proc_multicsp(epo,4);
    proc=struct('memo','csp_w');
    
    proc.train= ['[fv,csp_w]=  proc_multicsp(fv, 4); ' ...
        'fv= proc_variance(fv); ' ...
        'fv= proc_logarithm(fv);'];
    
    proc.apply= ['fv= proc_linearDerivation(fv, csp_w); ','fv= proc_variance(fv); ' ,'fv= proc_logarithm(fv);'];
%% CLASSIFIER
    
    [C_eeg, loss_eeg_std, out_eeg.out, memo] = xvalidation(epo,'RLDAshrink','proc',proc, 'kfold', 5);
    Result=(1-C_eeg)*100;
    Result_Std=loss_eeg_std;
    
    %All_csp_w(:,:,i)=csp_w;
    
    confu_matrix=confusionMatrix(epo.y,out_eeg.out);