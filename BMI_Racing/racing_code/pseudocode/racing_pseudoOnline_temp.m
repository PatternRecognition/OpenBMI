function [cf_out]=racing_pseudoOnline_temp(cnt,varargin)
 
duration=4;

opt=opt_cellToStruct(varargin{:});

LDA=opt.classifier(:);
CSP=opt.feature;
LDA(:,2)={'right vs left';'right vs foot';'left vs foot';...
    'right vs rest';'left vs rest';'foot vs rest';...
    'right vs others';'left vs others';'foot vs others';'rest vs others'};

if ~isfield(opt,'clf')
    opt.clf=1:size(LDA,1);
end

% marker= {'1','right';'2','left';'3','foot';'4','rest'};
% [EEG.data, EEG.marker, EEG.info]=Load_EEG(fullfile(fold, test),{'device','brainVision';'marker', marker;'fs', opt.fs});
% cnt=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, {'x','t','fs','y_dec','y_logic','y_class','class', 'chan'});
cls=unique(cnt.y_dec);
c={'m','c','y','k','g'};

%% 분류기 output과 true label
% classPair=size(LDA,1);
    figure()
for i=opt.clf
% define class
    cls_pair=strsplit(LDA{i,2});
    cls1=cls_pair{1};cls2=cls_pair{end};
    sprintf('%s vs %s',cls1,cls2)
    
    % Load data
    if strcmp(cls1,'others') || strcmp(cls2,'others')
        n_cls1=str2double(cnt.class(find(strcmp(cnt.class,{cls1}))-size(cnt.class,1)));
        n_cls2=5;
        CNT=changeLabels(cnt,{cls1,n_cls1;'others',n_cls2});
    else
        CNT=prep_selectClass(cnt,{'class',{cls1,cls2}});
        n_cls1=str2double(cnt.class(find(strcmp(cnt.class,{cls1}))-size(cnt.class,1)));
        n_cls2=str2double(cnt.class(find(strcmp(cnt.class,{cls2}))-size(cnt.class,1)));
    end
    
    % True class
    True=zeros(1,length(CNT.x));
    cls1_idx=round(cnt.t(cnt.y_dec==n_cls1));
    for k=1:length(cls1_idx)
        True(cls1_idx(k):cls1_idx(k)+(opt.fs*duration-1))=-1;
    end
    if n_cls2~=5
        cls2_idx=round(cnt.t(cnt.y_dec==n_cls2));
    else
        cls2_idx=round(cnt.t(cnt.y_dec~=n_cls1));
    end
    for k=1:length(cls2_idx)
        True(cls2_idx(k):cls2_idx(k)+(opt.fs*duration-1))=1;
    end
    
    % Time point
    bufferSize=opt.fs*opt.bufferSize;
    windowSize=opt.fs*opt.windowSize;
    stepSize=opt.fs*opt.stepSize;
    
    t=1:stepSize:length(CNT.x);
    t=t+bufferSize-1;
    t(t>length(CNT.x))=[];
%     t(t+bufferSize>length(CNT.x))=[];
%     t(1)=[];
%     cf_out=zeros(size(t));
    
    
    % pseudo-online
%     for j=1:(length(CNT.x)-bufferSize)/stepSize
%   CNT=prep_filter(CNT, {'frequency',opt.band;'fs',opt.fs });
    for j=1:length(t)
        Dat=CNT.x((stepSize*(j-1)+1):(stepSize*(j-1)+bufferSize),:);
        fDat=prep_filter(Dat, {'frequency',opt.band;'fs',opt.fs });
        fDat2=fDat(end-windowSize:end,:);
%         tm=func_projection(fDat2, CSP{i});
        ft=func_featureExtraction(fDat2, {'feature','logvar'});
        [cf_out(i,j)]=func_predict(ft, LDA{i,1});
        
        TIME=t(j);
        switch True(TIME)
            case -1
                CLASS=cls1;
            case 1
                CLASS=cls2;
            case 0
                CLASS='-';
        end
        if cf_out(i,j)<0, cf_output=cls1;else cf_output=cls2;end
%         sprintf('Time  : %.1f ~ %.1f (s)\nTrue  : %s\nOutput: %s\n',TIME/opt.fs,(TIME+bufferSize)/opt.fs,CLASS,cf_output)
    end
    
    TRUE_LB(:,i)=True(t);
    subplot(length(opt.clf),1,find(opt.clf==i))
    plot(t,cf_out(i,:))
    hold on
    
    % shadowing MI range
    if n_cls2==5
        st=CNT.t(CNT.y_dec==cls(n_cls1));
        for k=1:length(st)
            p=patch([st(k) st(k)+duration*opt.fs st(k)+duration*opt.fs st(k)],[-100 -100 100 100],c{n_cls1});
            set(p,'FaceAlpha',0.3);
        end
        st=CNT.t(CNT.y_dec==5);
        for k=1:length(st)
            p=patch([st(k) st(k)+duration*opt.fs st(k)+duration*opt.fs st(k)],[-100 -100 100 100],c{5});
            set(p,'FaceAlpha',0.3);
        end
    else
        for j=[n_cls1,n_cls2]
            st=CNT.t(CNT.y_dec==cls(j));
            for k=1:length(st)
                p=patch([st(k) st(k)+duration*opt.fs st(k)+duration*opt.fs st(k)],[-100 -100 100 100],c{j});
                set(p,'FaceAlpha',0.3);
            end
        end
    end
    plot(t,zeros(size(t)),'k')

    ylim([1.1*min(cf_out(i,:)),1.1*max(cf_out(i,:))])
    title([cls1,'(-) vs ',cls2,'(+)'])
end
