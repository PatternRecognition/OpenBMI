function [loss,OX,LABEL,CF_OUT]=racing_pseudoOnline3_temp(cnt,varargin)

% 20160726
% OVR 분류기 결과와 true label
% bufferSize, windowSize, stepSize

%% Data load
duration=4;
opt=opt_cellToStruct(varargin{:});
LDA=opt.classifier;
CSP=opt.feature;
if ~isfield(opt,'clf')
    opt.clf=1:size(LDA,1);
end
if ~isfield(opt,'topo')
    opt.topo=0;
end
% marker= {'1','right';'2','left';'3','foot';'4','rest'};
% [EEG.data, EEG.marker, EEG.info]=Load_EEG(fullfile(fold, test),{'device','brainVision';'marker', marker;'fs', opt.fs});
% cnt=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, {'x','t','fs','y_dec','y_logic','y_class','class', 'chan'});
cnt_temp=prep_selectClass(cnt,{'class',{'right', 'left','foot','rest'}});
SMT_temp=prep_segmentation(cnt_temp, {'interval',[-2000 5000]});

%% Basic setting
bufferSize=opt.fs*opt.bufferSize;
windowSize=opt.fs*opt.windowSize;
stepSize=opt.fs*opt.stepSize;
time_interval=1+0:bufferSize;

%% Topoplot setting
if opt.topo
MNT = opt_getMontage(SMT_temp);
center = [0 0];
theta = linspace(0,2*pi,360);
x = cos(theta)+center(1);
y = sin(theta)+center(2);
oldUnit = get(gcf,'units');
set(gcf,'units','normalized');
H = struct('ax', gca);
set(gcf,'CurrentAxes',H.ax);
tic
xe_org = MNT.x';
ye_org = MNT.y';
resolution = 100;
maxrad = max(1,max(max(abs(MNT.x)),max(abs(MNT.y))));
xx = linspace(-maxrad, maxrad, resolution);
yy = linspace(-maxrad, maxrad, resolution)';
end

CLASS_NAME={'non','right','left','foot','rest'};
acc=0;n=0;
LABEL=zeros(1,10000);
CF_OUT=zeros(1,10000);
%%
k=1;l=1;
while time_interval(end)<=length(cnt.x)
    cf_out=zeros(size(opt.clf));
    
    % true label
    ttt=time_interval(end)-windowSize;
    if k>length(cnt.t)
        break
    end
    if cnt.t(k)<ttt && ttt<cnt.t(k)+opt.fs*duration;
        LABEL(l)=cnt.y_dec(k);n=n+1;
    elseif ttt>=cnt.t(k)+opt.fs*duration
        LABEL(l)=0;k=k+1;
    else
        LABEL(l)=0;
    end
    Dat=cnt.x(time_interval,:);
    fDat=prep_filter(Dat, {'frequency',opt.band;'fs',opt.fs });
    fDat2.x=fDat(end-windowSize:end,:);
    fDat2.fs = opt.fs;
    % classifier output
    for i=opt.clf
        tm=func_projection(fDat2.x, CSP{i});
        ft=func_featureExtraction(fDat2, {'feature','logvar'});
        [cf_out(opt.clf==i)]=func_predict(ft, LDA{i});
    end
    CF_OUT(l)=find(cf_out==min(cf_out));
    time_interval=time_interval+stepSize;
    
    str2 = CLASS_NAME{LABEL(l)+1};
    str3 = CLASS_NAME{CF_OUT(l)+1};
    if strcmp(str2,str3)
        acc=acc+1;
    end
    
    % topo
    if opt.topo
    tmp3 = prep_envelope(fDat2);
    tm1=mean(tmp3.x(:,:));
    visual_topoplot(tm1, xe_org, ye_org, xx, yy);
    title(['true Label:  ' , str2, '   output classifier :  ', str3])
    drawnow;
    end
    l=l+1;
end

a=length(find(CF_OUT));
CF_OUT(a+1:end)=[];
LABEL(a+1:end)=[];
OX= CF_OUT==LABEL;
loss=acc/n;