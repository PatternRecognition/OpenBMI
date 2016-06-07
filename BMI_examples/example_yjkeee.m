clear all;
OpenBMI('E:\OpenBMI') % Edit the variable BMI if necessary
global BMI;
BMI.EEG_DIR=['E:\OpenBMI\BMI_data\RawEEG'];

%% DATA LOAD MODULE
file=fullfile(BMI.EEG_DIR, '\calibration_motorimageryVPkg');
marker={'1','left';'2','right';'3','foot';'4','rest'};
[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', [100]});

%% if you can redefine the marker information after Load_EEG function 
%% use  [marker, markerOrigin]=prep_defineClass(EEG.marke, marker)

field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
CNT=prep_selectClass(CNT,{'class',{'right', 'left'}});

%% PRE-PROCESSING MODULE
CNT=prep_filter(CNT, {'frequency', [0.5 40]});
SMT=prep_segmentation(CNT, {'interval', [750 3500]});
SMT.x=SMT.x(:,:,25:29)

%%%%%%%%%%%%%%%%%%%

%% DATA LOAD MODULE
file=fullfile(BMI.EEG_DIR, '\feedback_motorimageryVPkg');
marker={'1','left';'2','right';'3','foot';'4','rest'};
[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', [100]});

%% if you can redefine the marker information after Load_EEG function 
%% use  [marker, markerOrigin]=prep_defineClass(EEG.marke, marker)

field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
FNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
FNT=prep_selectClass(FNT,{'class',{'right', 'left'}});
FNT=prep_filter(FNT, {'frequency', [0.5 40]});
FNT=prep_segmentation(FNT, {'interval', [750 3500]});
FNT.x=FNT.x(:,:,25:29);


% %% train, test segmentation
% A=10;
% trial=50;
% 
% SMT1=SMT;
% SMT1.x=SMT.x(:,1:A,:);
% SMT1.t=SMT.t(:,1:A,:);
% SMT1.y_logic=SMT.y_logic(:,1:A);
% SMT1.y_dec=SMT.y_dec(1:A);
% SMT1.y_class=SMT.y_class(1,1:A);
% 
% SMT2=SMT;
% SMT2.x=SMT.x(:,A+1:trial,:);
% SMT2.t=SMT.t(:,A+1:trial,:);
% SMT2.y_logic=SMT.y_logic(:,A+1:trial);
% SMT2.y_dec=SMT.y_dec(A+1:trial);
% SMT2.y_class=SMT.y_class(1,A+1:trial);

loss=[];
            %% feature-extraction
            % feature=func_ar(SMT1,6,{'method','arburg'});
            train_feature=func_aar(SMT,{'Mode',[1 2]});
            test_feature=func_aar(FNT,{'Mode',[1 2]});
            
            %% CLASSIFIER MODULE
            [CF_PARAM]=func_train(train_feature,{'classifier','LDA'});
            [cf_out]=func_predict(test_feature, CF_PARAM);
            [loss]=eval_calLoss(test_feature.y_dec, cf_out);






% loss=[];
% for a=1:14
%     for v=1:6
%         for model=1:7
%             %% feature-extraction
%             % feature=func_ar(SMT1,6,{'method','arburg'});
%             train_feature=func_aar(SMT,[1,2],2);
%             test_feature=func_aar(FNT,[1,2],2);
%             
%             %% CLASSIFIER MODULE
%             [CF_PARAM]=func_train(train_feature,{'classifier','LDA'});
%             [cf_out]=func_predict(test_feature, CF_PARAM);
%             [loss(a,v,model)]=eval_calLoss(test_feature.y_dec, cf_out);
%             
%         end
%     end
% end




      %% SPATIAL-FREQUENCY OPTIMIZATION MODULE
            % [SMT, CSP_W, CSP_D]=func_csp(SMT,{'nPatterns', [3]});
            % FT=func_featureExtraction(SMT, {'feature','logvar'});
            
