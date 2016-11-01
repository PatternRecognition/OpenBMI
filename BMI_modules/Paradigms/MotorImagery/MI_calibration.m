function [LOSS, CSP, LDA] = MI_calibration( file ,band,fs)
%CAL_LOSS Summary of this function goes here
%   Detailed explanation goes here

marker={'1','left';'2','right';'3','foot'};
% marker= {'1','right';'2','left'};

[EEG.data, EEG.marker, EEG.info]=Load_EEG('C:\Vision\Raw Files\racingformontage',{'device','brainVision';'marker', marker;'fs', fs});

field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};

LOSS=cell(3,2);
CSP=cell(3,2);
LDA=cell(3,2);
%% right vs left
CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
% CNT=prep_selectClass(CNT,{'class',{'right', 'left'}});
%% PRE-PROCESSING MODULE

filter=band;
CNT=prep_filter(CNT, {'frequency', filter});
CNT=prep_selectChannels(CNT, {'Index', [1:64]})
SMT=prep_segmentation(CNT, {'interval', [750 3500]});

REST=prep_segmentation(CNT, {'interval', [-2750 0]});
REST.class={'4', 'rest'}.

out=prep_addTrials(SMT, REST)

set{1}={'rest', 'right'};
set{2}={'rest', 'left'};
set{3}={'rest', 'foot'};
set{4}={'rest', 'others'};
for i=1:4


% visuspect = visual_spectrum(SMT2 , {'Xaxis' , 'Frequency'; 'Yaxis' , 'Channel'});


%% ERD plot
% SMT1=prep_segmentation(CNT, {'interval', [-1000 3500]});
% 
% 
% % SMT is band-pass filtered signal
% 
% smt_env=prep_envelope(SMT1,{'Time',200});
% smt_base=prep_baseline(smt_env,{'Time',[-1000 0]});
% smt_avg=prep_average(smt_base);
% 
% figure()
% subplot(1,2,1)
% plot(smt_avg.x(:,1,find(strcmp(smt_avg.chan,'C3'))))
% hold on
% plot(smt_avg.x(:,2,find(strcmp(smt_avg.chan,'C3'))))
% title('C3')
% legend('right','left')
% 
% subplot(1,2,2)
% plot(smt_avg.x(:,1,find(strcmp(smt_avg.chan,'C4'))))
% hold on
% plot(smt_avg.x(:,2,find(strcmp(smt_avg.chan,'C4'))))
% title('C4')
% legend('right','left')
% 


%% SPATIAL-FREQUENCY OPTIMIZATION MODULE
[SMT, CSP_W, CSP_D]=func_csp(SMT,{'nPatterns', [3]});
FT=func_featureExtraction(SMT, {'feature','logvar'});

%% CLASSIFIER MODULE
[CF_PARAM]=func_train(FT,{'classifier','LDA'});


CV.prep={ % commoly applied to training and test data before data split
    'CNT=prep_filter(CNT, {"frequency", [11 17]})'
    'SMT=prep_segmentation(CNT, {"interval", [750 3500]})'
    };
CV.train={
    '[SMT, CSP_W, CSP_D]=func_csp(SMT,{"nPatterns", [3]})'
    'FT=func_featureExtraction(SMT, {"feature","logvar"})'
    '[CF_PARAM]=func_train(FT,{"classifier","LDA"})'
    };
CV.test={
    'SMT=func_projection(SMT, CSP_W)'
    'FT=func_featureExtraction(SMT, {"feature","logvar"})'
    '[cf_out]=func_predict(FT, CF_PARAM)'
    };
CV.option={
'KFold','5'
% 'leaveout'
};

[loss]=eval_crossValidation(CNT, CV); % input : eeg, or eeg_epo
CSP{1,1}=CSP_W; CSP{1,2}='right vs left';
LDA{1,1}=CF_PARAM; LDA{1,2}='right vs left';
LOSS{1,1}=loss;LOSS{1,2}='right vs left';


%% right vs foot
CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
CNT=prep_selectClass(CNT,{'class',{'right', 'foot'}});
%% PRE-PROCESSING MODULE
filter=band;
CNT=prep_filter(CNT, {'frequency', filter});
SMT=prep_segmentation(CNT, {'interval', [750 3500]});
% % visuspect = visual_spectrum(SMT , {'Xaxis' , 'Frequency'; 'Yaxis' , 'Channel'});
% 
% 
% %% ERD plot
% % SMT is band-pass filtered signal
% 
% smt_env=prep_envelope(SMT1,{'Time',200});
% smt_base=prep_baseline(smt_env,{'Time',[-1000 0]});
% smt_avg=prep_average(smt_base);
% 
% figure()
% subplot(1,2,1)
% plot(smt_avg.x(:,1,find(strcmp(smt_avg.chan,'C3'))))
% hold on
% plot(smt_avg.x(:,2,find(strcmp(smt_avg.chan,'C3'))))
% title('C3')
% legend('right','foot')
% 
% subplot(1,2,2)
% plot(smt_avg.x(:,1,find(strcmp(smt_avg.chan,'Cz'))))
% hold on
% plot(smt_avg.x(:,2,find(strcmp(smt_avg.chan,'Cz'))))
% title('Cz')
% legend('right','foot')
% 
% 
% 
% 
% 
%% SPATIAL-FREQUENCY OPTIMIZATION MODULE
[SMT, CSP_W, CSP_D]=func_csp(SMT,{'nPatterns', [3]});
FT=func_featureExtraction(SMT, {'feature','logvar'});

%% CLASSIFIER MODULE
[CF_PARAM]=func_train(FT,{'classifier','LDA'});


CV.prep={ % commoly applied to training and test data before data split
    'CNT=prep_filter(CNT, {"frequency", [7 13]})'
    'SMT=prep_segmentation(CNT, {"interval", [750 3500]})'
    };
CV.train={
    '[SMT, CSP_W, CSP_D]=func_csp(SMT,{"nPatterns", [3]})'
    'FT=func_featureExtraction(SMT, {"feature","logvar"})'
    '[CF_PARAM]=func_train(FT,{"classifier","LDA"})'
    };
CV.test={
    'SMT=func_projection(SMT, CSP_W)'
    'FT=func_featureExtraction(SMT, {"feature","logvar"})'
    '[cf_out]=func_predict(FT, CF_PARAM)'
    };
CV.option={
'KFold','5'
% 'leaveout'
};

[loss]=eval_crossValidation(CNT, CV); % input : eeg, or eeg_epo

CSP{2,1}=CSP_W; CSP{2,2}='right vs foot';
LDA{2,1}=CF_PARAM; LDA{2,2}='right vs foot';
LOSS{2,1}=loss; LOSS{2,2}='right vs foot';
%% left vs foot

CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
CNT=prep_selectClass(CNT,{'class',{'left', 'foot'}});
%% PRE-PROCESSING MODULE
filter=[7 13];
CNT=prep_filter(CNT, {'frequency', filter});
SMT=prep_segmentation(CNT, {'interval', [750 3500]});

% 
% %% ERD plot
% % SMT is band-pass filtered signal
% 
% smt_env=prep_envelope(SMT1,{'Time',200});
% smt_base=prep_baseline(smt_env,{'Time',[-1000 0]});
% smt_avg=prep_average(smt_base);
% 
% figure()
% subplot(1,2,1)
% plot(smt_avg.x(:,1,find(strcmp(smt_avg.chan,'C4'))))
% hold on
% plot(smt_avg.x(:,2,find(strcmp(smt_avg.chan,'C4'))))
% title('C4')
% legend('left','foot')
% 
% subplot(1,2,2)
% plot(smt_avg.x(:,1,find(strcmp(smt_avg.chan,'Cz'))))
% hold on
% plot(smt_avg.x(:,2,find(strcmp(smt_avg.chan,'Cz'))))
% title('Cz')
% legend('left','foot')
% 
% 
% 

%% SPATIAL-FREQUENCY OPTIMIZATION MODULE
[SMT, CSP_W, CSP_D]=func_csp(SMT,{'nPatterns', [3]});
FT=func_featureExtraction(SMT, {'feature','logvar'});

%% CLASSIFIER MODULE
[CF_PARAM]=func_train(FT,{'classifier','LDA'});



CV.prep={ % commoly applied to training and test data before data split
    'CNT=prep_filter(CNT, {"frequency", [7 13]})'
    'SMT=prep_segmentation(CNT, {"interval", [750 3500]})'
    };
CV.train={
    '[SMT, CSP_W, CSP_D]=func_csp(SMT,{"nPatterns", [3]})'
    'FT=func_featureExtraction(SMT, {"feature","logvar"})'
    '[CF_PARAM]=func_train(FT,{"classifier","LDA"})'
    };
CV.test={
    'SMT=func_projection(SMT, CSP_W)'
    'FT=func_featureExtraction(SMT, {"feature","logvar"})'
    '[cf_out]=func_predict(FT, CF_PARAM)'
    };
CV.option={
'KFold','5'
% 'leaveout'
};

[loss]=eval_crossValidation(CNT, CV); % input : eeg, or eeg_epo

CSP{3,1}=CSP_W; CSP{3,2}='left vs foot';
LDA{3,1}=CF_PARAM; LDA{3,2}='left vs foot';
LOSS{3,1}=loss;LOSS{3,2}='left vs foot';
end

