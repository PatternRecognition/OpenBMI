function [CSP_W, CF_PARAM, loss] = racing_calibration_( file ,band, fs)
%CAL_LOSS Summary of this function goes here
%   Detailed explanation goes here

marker={'1','left';'2','right';'3','foot'; '4', 'rest'};
[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', fs});
field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
mCNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
mCNT2=prep_filter(mCNT, {'frequency', band});

% Testing Jihoon
% mCNT2= opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
%
%% binary classifier
set{1}={'binary';{'right', 'left'}};
set{2}={'binary';{'right', 'foot'}};
set{3}={'binary';{'left', 'foot'}};
set{4}={'binary';{'right', 'rest'}};
set{5}={'binary';{'left', 'rest'}};
set{6}={'binary';{'foot', 'rest'}};
set{7}={'ovr'; {'right', 'others'}};
set{8}={'ovr'; {'left', 'others'}};
set{9}={'ovr'; {'foot', 'others'}};
set{10}={'ovr'; {'rest', 'others'}};

out=changeLabels(mCNT,{'right',1;'others',2});

for i=1:length(set)
    
    if strcmp(set{1,i}{1,1},'binary')
        CNT=prep_selectClass(mCNT,{'class',{set{1,i}{2,1}{:}}});
        filtered_CNT=prep_selectClass(mCNT2,{'class',{set{1,i}{2,1}{:}}});
    else % 
        CNT=changeLabels(mCNT,{set{1,i}{2,1}{1},1;'others',2});     
        filtered_CNT=changeLabels(mCNT2,{set{1,i}{2,1}{1},1;'others',2});
    end
    SMT=prep_segmentation(CNT, {'interval', [750 3500]});
    filtered_SMT=prep_segmentation(filtered_CNT, {'interval', [750 3500]});
    SMT = prep_resample(SMT, 100);
    filtered_SMT = prep_resample(filtered_SMT, 100); 
    chinx={};
    chinx{1} = find(strcmp(filtered_SMT.chan , 'C3') == 1);
    chinx{2} = find(strcmp(filtered_SMT.chan , 'Cz') == 1);
    chinx{3} = find(strcmp(filtered_SMT.chan , 'C4') == 1);
    SMT2 = prep_envelope(filtered_SMT);
    SMT_class1 = prep_selectClasses_(SMT2 , {'Class' , SMT2.class{1,2}});
    SMT_class2 = prep_selectClasses_(SMT2 , {'Class' , SMT2.class{2,2}});
    SMT_env1 = mean(SMT_class1.x,2);SMT_env2 = mean(SMT_class2.x,2);
    SMT_env1 = squeeze(SMT_env1);SMT_env2 = squeeze(SMT_env2);
    figure(1)
%     figure(2*i-1)
    a = [1 2 3;4 5 6;7 8 9;10 11 12;13 14 15;16 17 18;19 20 21;22 23 24; 25 26 27;28 29 30];
    subplot(10,3, a(i,1))
    plot(SMT_env1(:,chinx{1}));title('C3');hold on;
    plot(SMT_env2(:,chinx{1}) , 'r');
    subplot(10,3,a(i,2))
    plot(SMT_env1(:,chinx{2}));title('Cz');hold on;
    plot(SMT_env2(:,chinx{2}),'r');
    subplot(10,3,a(i,3))
    plot(SMT_env1(:,chinx{3}));title('C4');hold on;
    plot(SMT_env2(:,chinx{3}),'r');
    
    figure(2)
    subplot(2,5,i)
    visuspect = visual_spectrum_BCIracing_(SMT , {'Xaxis' , 'Frequency'; 'Yaxis' , 'Channel'});
    
end
%     [SMT, CSP_W{i}, CSP_D]=func_csp(SMT,{'nPatterns', [3]});
%     FT=func_featureExtraction(SMT, {'feature','logvar'});
%     [CF_PARAM{i}]=func_train(FT,{'classifier','LDA'});
%     
%     CV.var.band=band;
%     CV.var.interval=[750 3500];
%     CV.prep={ % commoly applied to training and test data before data split
%         'CNT=prep_filter(CNT, {"frequency", band})'
%         'SMT=prep_segmentation(CNT, {"interval", interval})'
%         };
%     CV.train={
%         '[SMT, CSP_W, CSP_D]=func_csp(SMT,{"nPatterns", [3]})'
%         'FT=func_featureExtraction(SMT, {"feature","logvar"})'
%         '[CF_PARAM]=func_train(FT,{"classifier","LDA"})'
%         };
%     CV.test={
%         'SMT=func_projection(SMT, CSP_W)'
%         'FT=func_featureExtraction(SMT, {"feature","logvar"})'
%         '[cf_out]=func_predict(FT, CF_PARAM)'
%         };
%     CV.option={
%         'KFold','5'
%         % 'leaveout'
%         };
%     
%     [loss{i}]=eval_crossValidation(SMT, CV); % input : eeg, or eeg_epo
% end



% CNT=prep_selectClass(CNT,{'class',{'right', 'left'}});


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

% 
% %% SPATIAL-FREQUENCY OPTIMIZATION MODULE
% [SMT, CSP_W, CSP_D]=func_csp(SMT,{'nPatterns', [3]});
% FT=func_featureExtraction(SMT, {'feature','logvar'});
% 
% %% CLASSIFIER MODULE
% [CF_PARAM]=func_train(FT,{'classifier','LDA'});
% 
% 
% CV.prep={ % commoly applied to training and test data before data split
%     'CNT=prep_filter(CNT, {"frequency", [11 17]})'
%     'SMT=prep_segmentation(CNT, {"interval", [750 3500]})'
%     };
% CV.train={
%     '[SMT, CSP_W, CSP_D]=func_csp(SMT,{"nPatterns", [3]})'
%     'FT=func_featureExtraction(SMT, {"feature","logvar"})'
%     '[CF_PARAM]=func_train(FT,{"classifier","LDA"})'
%     };
% CV.test={
%     'SMT=func_projection(SMT, CSP_W)'
%     'FT=func_featureExtraction(SMT, {"feature","logvar"})'
%     '[cf_out]=func_predict(FT, CF_PARAM)'
%     };
% CV.option={
% 'KFold','5'
% % 'leaveout'
% };
% 
% [loss]=eval_crossValidation(CNT, CV); % input : eeg, or eeg_epo
% CSP{1,1}=CSP_W; CSP{1,2}='right vs left';
% LDA{1,1}=CF_PARAM; LDA{1,2}='right vs left';
% LOSS{1,1}=loss;LOSS{1,2}='right vs left';
% 
% 
% %% right vs foot
% CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
% CNT=prep_selectClass(CNT,{'class',{'right', 'foot'}});
% %% PRE-PROCESSING MODULE
% filter=band;
% CNT=prep_filter(CNT, {'frequency', filter});
% SMT=prep_segmentation(CNT, {'interval', [750 3500]});
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
% %% SPATIAL-FREQUENCY OPTIMIZATION MODULE
% [SMT, CSP_W, CSP_D]=func_csp(SMT,{'nPatterns', [3]});
% FT=func_featureExtraction(SMT, {'feature','logvar'});
% 
% %% CLASSIFIER MODULE
% [CF_PARAM]=func_train(FT,{'classifier','LDA'});
% 
% 
% CV.prep={ % commoly applied to training and test data before data split
%     'CNT=prep_filter(CNT, {"frequency", [7 13]})'
%     'SMT=prep_segmentation(CNT, {"interval", [750 3500]})'
%     };
% CV.train={
%     '[SMT, CSP_W, CSP_D]=func_csp(SMT,{"nPatterns", [3]})'
%     'FT=func_featureExtraction(SMT, {"feature","logvar"})'
%     '[CF_PARAM]=func_train(FT,{"classifier","LDA"})'
%     };
% CV.test={
%     'SMT=func_projection(SMT, CSP_W)'
%     'FT=func_featureExtraction(SMT, {"feature","logvar"})'
%     '[cf_out]=func_predict(FT, CF_PARAM)'
%     };
% CV.option={
% 'KFold','5'
% % 'leaveout'
% };
% 
% [loss]=eval_crossValidation(CNT, CV); % input : eeg, or eeg_epo
% 
% CSP{2,1}=CSP_W; CSP{2,2}='right vs foot';
% LDA{2,1}=CF_PARAM; LDA{2,2}='right vs foot';
% LOSS{2,1}=loss; LOSS{2,2}='right vs foot';
% %% left vs foot
% 
% CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
% CNT=prep_selectClass(CNT,{'class',{'left', 'foot'}});
% %% PRE-PROCESSING MODULE
% filter=[7 13];
% CNT=prep_filter(CNT, {'frequency', filter});
% SMT=prep_segmentation(CNT, {'interval', [750 3500]});

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
% % 
% 
% %% SPATIAL-FREQUENCY OPTIMIZATION MODULE
% [SMT, CSP_W, CSP_D]=func_csp(SMT,{'nPatterns', [3]});
% FT=func_featureExtraction(SMT, {'feature','logvar'});
% 
% %% CLASSIFIER MODULE
% [CF_PARAM]=func_train(FT,{'classifier','LDA'});
% 
% 
% 
% CV.prep={ % commoly applied to training and test data before data split
%     'CNT=prep_filter(CNT, {"frequency", [7 13]})'
%     'SMT=prep_segmentation(CNT, {"interval", [750 3500]})'
%     };
% CV.train={
%     '[SMT, CSP_W, CSP_D]=func_csp(SMT,{"nPatterns", [3]})'
%     'FT=func_featureExtraction(SMT, {"feature","logvar"})'
%     '[CF_PARAM]=func_train(FT,{"classifier","LDA"})'
%     };
% CV.test={
%     'SMT=func_projection(SMT, CSP_W)'
%     'FT=func_featureExtraction(SMT, {"feature","logvar"})'
%     '[cf_out]=func_predict(FT, CF_PARAM)'
%     };
% CV.option={
% 'KFold','5'
% % 'leaveout'
% };
% 
% [loss]=eval_crossValidation(CNT, CV); % input : eeg, or eeg_epo
% 
% CSP{3,1}=CSP_W; CSP{3,2}='left vs foot';
% LDA{3,1}=CF_PARAM; LDA{3,2}='left vs foot';
% LOSS{3,1}=loss;LOSS{3,2}='left vs foot';
end

