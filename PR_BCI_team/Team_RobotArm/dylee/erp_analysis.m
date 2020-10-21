%% new erp

clc; close all; clear all;

%% file
dd='C:\Users\Doyeunlee\Desktop\Analysis\rawdata\';

% reaching
% filelist={'jmlee_reaching_MI'};
% filelist={'eslee_reaching_MI'};
filelist={'dslim_reaching_realMove'};

% selectChannels = [11 12 13 43 44 45 15 16 17 18 47 48 49 20 21 22 23 51 52 53];
% selectChannels = [1:64];
selectChannels = [18];



for i = 1:length(filelist)
    [cnt, mrk, mnt] = eegfile_loadMatlab([dd filelist{i}]);
    
    band = [8 12];
    
    [b, a] = butter(5, band/cnt.fs*2);
    cnt_flt = proc_channelwise(cnt, 'filtfilt', b, a);
    
    ival = [-500 3000];
    epo = cntToEpo(cnt_flt, mrk, ival);
    
    fv = proc_rectifyChannels(epo);
    fv = proc_movingAverage(fv, 200, 'centered');
    fv = proc_baseline(fv, [-500 0]);
    
    epo = proc_selectClasses(epo, {'Forward','Backward','Right','Left','Up','Down'});
    epoUp=proc_selectClasses(epo,{'Up'});
    
    epoUp.x=epoUp.x(:,:,1);
    clab = epo.clab;
    
    classes=size(epo.className,2);
    trial=50;
    
    for ii=1:classes
        if strcmp(epo.className{ii},'Rest')
            epoRest=proc_selectClasses(epo,{epo.className{ii}});
            epoRest.x=datasample(epoRest.x,trial,3,'Replace',false);
            epoRest.y=datasample(epoRest.y,trial,2,'Replace',false);
        else
            epo_check(ii)=proc_selectClasses(epo,{epo.className{ii}});
            epo_check(ii).x=datasample(epo_check(ii).x,trial,3,'Replace',false);
            epo_check(ii).y=datasample(epo_check(ii).y,trial,2,'Replace',false);
        end
    end
    if classes<6
        epo_check(size(epo_check,2)+1)=epoRest;
    end
    %concatenate the classes
    for ii=1:size(epo_check,2)
        if ii==1
            concatEpo=epo_check(ii);
        else
            concatEpo=proc_appendEpochs(concatEpo, epo_check(ii));
        end
    end
    
    %% CSP - feature extraction
    [csp_fv,csp_w,csp_eig]=proc_multicsp(concatEpo,3);
    proc=struct('memo','csp_w');
    
    proc.train=['[fv,csp_w]= proc_multicsp(fv, 3);' ...
        'fv=proc_variance(fv);' ...
        'fv=proc_logarithm(fv);'];
    
    proc.apply=['fv=proc_linearDerivation(fv, csp_w);','fv=proc_variance(fv);','fv=proc_logarithm(fv);'];
    
    %% LDA - Classifier
    [C_eeg, loss_eeg_std, out_4eg.out, memo] = xvalidation(concatEpo, 'RLDAshrink','proc',proc,'kfold',5);
    Result(i)=1-C_eeg;
    Result_Std(i)=loss_eeg_std;
    N=Result(i);
    All_csp_w(:,:,i)=csp_w;
end


plot_channel(epoUp, clab);


%% erp visualization
% 가로축 시간 세로축 주파수 색깔 dB
% band = [0 10];
% [dat] = proc_specgram(epo, band, selectChannels);
% tLim = [0 300000];
% yLim = [0 50]; %dB
% figure (1)
% showSpecgramHead(dat, mnt, band, tLim, yLim);

% saveas(fig, 'dslim_reaching.png');




