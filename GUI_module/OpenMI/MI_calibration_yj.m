function [LOSS, CSP, LDA] = MI_calibration_yj(EEG,band,fs,interval,varargin)

opt=opt_cellToStruct(varargin{:});
cls=opt.nClass;
% switch cls
%     case 2
%         marker={'1','right';'2','left'};
%     case 3
%         marker={'1','right';'2','left';'3','foot'};
%     case 4
%         marker={'1','right';'2','left';'3','foot';'4','rest'};
% end
field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
% [EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker',marker;'fs',fs});
CNT_=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);

if ~isfield(opt,'channel'),opt.channel=1:EEG.data.nCh;end
if ~isfield(opt,'erd'),opt.erd=0;end

% cross validation parameters
CV.var.band=band;
CV.var.interval=interval;
CV.var.fv='logvar';        % temp
CV.var.classfier='LDA';    % temp
CV.var.evaluation='KFold'; % temp
CV.var.k=5;                % temp
CV.prep={
    'CNT=prep_filter(CNT, {"frequency", band})'
    'SMT=prep_segmentation(CNT, {"interval", interval})'
    };
CV.train={
    '[SMT, CSP_W, CSP_D]=func_csp(SMT,{"nPatterns", [3]})'
    'FT=func_featureExtraction(SMT, {"feature",fv})'
    '[CF_PARAM]=func_train(FT,{"classifier",classfier})'
    };
CV.test={
    'SMT=func_projection(SMT, CSP_W)'
    'FT=func_featureExtraction(SMT, {"feature",fv})'
    '[cf_out]=func_predict(FT, CF_PARAM)'
    };
CV.option={
    'evaluation' , 'k'
    };

n=size(CNT_.class,1);
pairIndex=[];
for i=1:(n-1)
    pairIndex=[pairIndex,[1:i;(1:i)+(n-i)]];
end
[~,I]=sort(sum(pairIndex));
pairIndex=pairIndex(:,I);
pairClass=cell(size(pairIndex));
for i=1:length(pairClass(:))
    pairClass{i}=CNT_.class(pairIndex(i),2);
end
pairNum=size(pairIndex,2);
LOSS=cell(pairNum,2);
CSP=cell(pairNum,2);
LDA=cell(pairNum,2);

for pair=1:pairNum

    pairstr=sprintf('%s vs %s',char(pairClass{1,pair}),char(pairClass{2,pair}));
    disp(pairstr);
    
    CNT=prep_selectClass(CNT_,{'class',{pairClass{1,pair}, pairClass{2,pair}}});
    CNT=prep_selectChannels(CNT,{'Index',opt.channel});
    
    %% Pre-processing module
    CNT=prep_filter(CNT, {'frequency', band});
    SMT=prep_segmentation(CNT, {'interval', interval});
    
    %% Spatial-frequency optimization module
    [SMT, CSP_W, CSP_D]=func_csp(SMT,{'nPatterns', [3]});
    FT=func_featureExtraction(SMT, {'feature',CV.var.fv});
    
    %% Classifier module
    [CF_PARAM]=func_train(FT,{'classifier',CV.var.classfier});
    
    %% Evaluation
    [loss]=eval_crossValidation_(CNT, CV);
    
    CSP{pair,1}=CSP_W; CSP{pair,2}=pairstr;
    LDA{pair,1}=CF_PARAM; LDA{pair,2}=pairstr;
    LOSS{pair,1}=loss;LOSS{pair,2}=pairstr;
    
    
end
