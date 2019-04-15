clear all; clc;
startup_bbci_toolbox

%% add normalize
name={'1_bykim','2_dblee','3_eskim','4_jhsuh','6_mhlee','7_mjkim','8_oykwon','9_prchoi','10_smkang','11_spseo','12_yskim','13_jyha','14_hmyoo','15_jwlee','16_wbsim','17_hwnoh'}; % session 1

for sub=1:length(name)
    
    files={'mi_off','mi_on'};
    stimDef= {1, 2;
        'left', 'right'};
    
    session = {'session1'};
    
    for i=1:length(files)
        
        file3 = ['E:\Users\cvpr\Desktop\StarlabDB_2nd\',name{sub},'\'];
        BMI.EEG_DIR=['E:\Users\cvpr\Desktop\StarlabDB_2nd\',name(sub),'\',session];
        
        BMI.EEG_DIR=cell2mat(BMI.EEG_DIR);
        file=fullfile(BMI.EEG_DIR, files{i});
        
        [cnt, mrk_orig, hdr] = file_readBV(file, 'Fs', 100);
        % create mrk and mnt
        mrk= mrk_defineClasses(mrk_orig, stimDef);
        mrk.orig= mrk_orig;
        mnt= mnt_setElectrodePositions(cnt.clab);
        mnt= mnt_setGrid(mnt, 'M');
        
        % Apply highpass filter to reduce drifts
        b= procutil_firlsFilter(0.5, cnt.fs);
        cnt= proc_filtfilt(cnt, b);
        
        epo= proc_segmentation(cnt, mrk, [750 3500]);
        %         epo_{i}= proc_selectChannels(epo, [1:45 48 50 53 61:66]);
%         epo_{i}= proc_selectChannels(epo, [1:32]);
        epo_{i}= proc_selectChannels(epo, [8:11 13:15 18:21 33:38 39 41]);
        
        epo_rest= proc_segmentation(cnt, mrk, [-1000 0]);
        %         epo_r{i}= proc_selectChannels(epo_rest, [1:45 48 50 53 61:66]);
%         epo_r{i}= proc_selectChannels(epo_rest, [1:32]);
        epo_r{i}= proc_selectChannels(epo_rest, [8:11 13:15 18:21 33:38 39 41]);
    end
    
    
    train_epo=epo_{1};
    test_epo=epo_{2};
    
    train_epo_r=epo_r{1};
    test_epo_r=epo_r{2};
    
    % 일단 단일 주파수로
    base=[8 30];
    [b,a]= butter(5, [base(1) base(2)]/cnt.fs*2);
    train_epo=proc_filt(train_epo, b, a);
    test_epo=proc_filt(test_epo, b, a);
    
    train_epo_r=proc_filt(train_epo_r, b, a);
    test_epo_r=proc_filt(test_epo_r, b, a);
    
    [nD nCH nTri]=size(train_epo.x);
    
    
    for i=1:nTri
        train_cov(:,:,i) = cov(squeeze(train_epo.x(:,:,i)));
        test_cov(:,:,i) = cov(squeeze(test_epo.x(:,:,i)));
        
        train_cov_r(:,:,i) = cov(squeeze(train_epo_r.x(:,:,i)));
        test_cov_r(:,:,i) = cov(squeeze(test_epo_r.x(:,:,i)));
    end
    
    train_cov=train_cov;
    test_cov=test_cov;
    
    [w h tri] = size(train_cov);
    clear a b c C
    
    file33 = ['E:\Users\cvpr\Desktop\input\Deep_feature\cov_input_19ch\Cov_input_830_bbci\',name{sub},'\'];
    
    tm1=permute(train_cov, [3 1 2]);
    tm1=reshape(tm1, [tri w*h]);
    trX=tm1;
    a=find(train_epo.y(1,:)==1);
    b=find(train_epo.y(2,:)==1);
    c= zeros(length(train_epo.y),1)';
    c(a)=0;
    c(b)=1;
    C= c;
    trY=C;
    train = [trX trY'];
    clear C c a b 
    str = [file33 'trX.csv'];
    csvwrite(str, trX);
    str = [file33 'trY.csv'];
    csvwrite(str, trY);
    
    tm1=permute(test_cov, [3 1 2]);
    tm1=reshape(tm1, [tri w*h]);
    teX=tm1;
    a=find(test_epo.y(1,:)==1);
    b=find(test_epo.y(2,:)==1);
    c= zeros(length(test_epo.y),1)';
    c(a)=0;
    c(b)=1;
    C= c;
    teY=C;
    test= [teX teY'];
    str2 = [file33 'teX.csv'];
    csvwrite(str2, teX);
    str2 = [file33 'teY.csv'];
    csvwrite(str2, teY);
    
    clear C c a b 
        
%     filename5= ['trX.mat'];
%     filename6= ['trY.mat'];
%     filename7= ['teX.mat'];
%     filename8= ['teY.mat'];
%     file33 = ['E:\Users\cvpr\Desktop\input\New_input\Cov_input_813_bbci\',name{sub},'\'];
%     fname = fullfile ( file33, filename5 );
%     save ( fname, 'trX' );
%     fname = fullfile ( file33, filename6 );
%     save ( fname, 'trY' );
%     fname = fullfile ( file33, filename7 );
%     save ( fname, 'teX' );
%     fname = fullfile ( file33, filename8 );
%     save ( fname, 'teY' );
    
    
    clear a b base BMI BTB cnt epo epo_ epo_r epo_rest h hdr i mnt mrk mrk_orig nCH nD nTri test_cov test_cov_r test_epo test_epo_r teX teY tm1
    clear train_cov train_cov_r train_epo train_epo_r tri trX trY w
end

%% Opt. Freq. 

clear all; clc;
startup_bbci_toolbox

name={'1_bykim','2_dblee','3_eskim','4_jhsuh','6_mhlee','7_mjkim','8_oykwon','9_prchoi','10_smkang','11_spseo','12_yskim','13_jyha','14_hmyoo','15_jwlee','16_wbsim','17_hwnoh'}; % session 1

for sub=1:length(name)
    
    files={'mi_off','mi_on'};
    stimDef= {1, 2;
        'left', 'right'};
    
    session = {'session1'};
    
    for i=1:length(files)        
        file3 = ['E:\Users\cvpr\Desktop\StarlabDB_2nd\',name{sub},'\'];
        BMI.EEG_DIR=['E:\Users\cvpr\Desktop\StarlabDB_2nd\',name(sub),'\',session];
        
        BMI.EEG_DIR=cell2mat(BMI.EEG_DIR);
        file=fullfile(BMI.EEG_DIR, files{i});
        
        [cnt, mrk_orig, hdr] = file_readBV(file, 'Fs', 100);
        % create mrk and mnt
        mrk= mrk_defineClasses(mrk_orig, stimDef);
        mrk.orig= mrk_orig;
        mnt= mnt_setElectrodePositions(cnt.clab);
        mnt= mnt_setGrid(mnt, 'M');
        
        % Apply highpass filter to reduce drifts
        b= procutil_firlsFilter(0.5, cnt.fs);
        cnt= proc_filtfilt(cnt, b);
        
        epo= proc_segmentation(cnt, mrk, [750 3500]);
        epo_{i}= proc_selectChannels(epo, [1:45 48 50 53 61:66]);
        %         epo_{i}= proc_selectChannels(epo, [1:32]);
        %         epo_{i}= proc_selectChannels(epo, [8:11 13:15 18:21 33:38 39 41]);
        
        epo_rest= proc_segmentation(cnt, mrk, [-1000 0]);
        epo_r{i}= proc_selectChannels(epo_rest, [1:45 48 50 53 61:66]);
        %         epo_r{i}= proc_selectChannels(epo_rest, [1:32]);
        %         epo_r{i}= proc_selectChannels(epo_rest, [8:11 13:15 18:21 33:38 39 41]);
    end
        
    train_epo=epo_{1};
    test_epo=epo_{2};
    
    train_epo_r=epo_r{1};
    test_epo_r=epo_r{2};
    
    % 일단 단일 주파수로
    load A;
    base=A(sub,:);
%     base=[8 30];
    [b,a]= butter(5, [base(1) base(2)]/cnt.fs*2);
    train_epo_r=proc_filt(train_epo, b, a);
    test_epo=proc_filt(test_epo, b, a);
    clear base 
    
    train_epo_r=proc_filt(train_epo_r, b, a);
    test_epo_r=proc_filt(test_epo_r, b, a);
    
    [nD nCH nTri]=size(train_epo.x);
    
    [WW,train_epo1, CSP_W, CSP_A, SCORE]=for_proc_csp(train_epo);
    train_epo2=proc_linearDerivation(train_epo, WW);
    clear train_epo1 CSP_W CSP_A SCORE    
    test_epo2=proc_linearDerivation(test_epo, WW);
    clear WW test_epo1 CSP_W CSP_A SCORE 
    
    for i=1:nTri
        train_cov(:,:,i) = cov(squeeze(train_epo2.x(:,:,i)));
        test_cov(:,:,i) = cov(squeeze(test_epo2.x(:,:,i)));        
%         train_cov_r(:,:,i) = cov(squeeze(train_epo_r.x(:,:,i)));
%         test_cov_r(:,:,i) = cov(squeeze(test_epo_r.x(:,:,i)));
    end
    clear train_epo2 test_epo2
    
    train_cov=train_cov;
    test_cov=test_cov;
    
    [w h tri] = size(train_cov);
    clear a b c C
    
    file33 = ['E:\Users\cvpr\Desktop\input\Deep_feature\opti_csp_cov_input_54ch\',name{sub},'\'];
    
    tm1=permute(train_cov, [3 1 2]);
    tm1=reshape(tm1, [tri w*h]);
    trX=tm1;
    a=find(train_epo.y(1,:)==1);
    b=find(train_epo.y(2,:)==1);
    c= zeros(length(train_epo.y),1)';
    c(a)=0;
    c(b)=1;
    C= c;
    trY=C;
    train = [trX trY'];
    clear C c a b
    str = [file33 'trX.csv'];
    csvwrite(str, trX);
    str = [file33 'trY.csv'];
    csvwrite(str, trY);
    
    tm1=permute(test_cov, [3 1 2]);
    tm1=reshape(tm1, [tri w*h]);
    teX=tm1;
    a=find(test_epo.y(1,:)==1);
    b=find(test_epo.y(2,:)==1);
    c= zeros(length(test_epo.y),1)';
    c(a)=0;
    c(b)=1;
    C= c;
    teY=C;
    test= [teX teY'];
    str2 = [file33 'teX.csv'];
    csvwrite(str2, teX);
    str2 = [file33 'teY.csv'];
    csvwrite(str2, teY);
    
    clear C c a b
    
    %     filename5= ['trX.mat'];
    %     filename6= ['trY.mat'];
    %     filename7= ['teX.mat'];
    %     filename8= ['teY.mat'];
    %     file33 = ['E:\Users\cvpr\Desktop\input\New_input\Cov_input_813_bbci\',name{sub},'\'];
    %     fname = fullfile ( file33, filename5 );
    %     save ( fname, 'trX' );
    %     fname = fullfile ( file33, filename6 );
    %     save ( fname, 'trY' );
    %     fname = fullfile ( file33, filename7 );
    %     save ( fname, 'teX' );
    %     fname = fullfile ( file33, filename8 );
    %     save ( fname, 'teY' );
    
    
    clear a b base BMI BTB cnt epo epo_ epo_r epo_rest h hdr i mnt mrk mrk_orig nCH nD nTri test_cov test_cov_r test_epo test_epo_r teX teY tm1
    clear train_cov train_cov_r train_epo train_epo_r tri trX trY w
end







