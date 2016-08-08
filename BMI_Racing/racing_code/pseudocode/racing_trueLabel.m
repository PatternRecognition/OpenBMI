function [LABEL]=racing_trueLabel(BMI,test,varargin)
% example:
% [TRUE_LABEL]=racing_trueLabel(BMI,'\2016_07_12_mhlee_test1',{'fs',fs;...
%     'interval',interval;'bufferSize',5;'windowSize',3;'stepSize',0.5});
%
% bufferSize, windowSize, stepSize (s, not ms)

duration=4;

opt=opt_cellToStruct(varargin{:});
marker= {'1','right';'2','left';'3','foot';'4','rest'};
[EEG.data,EEG.marker,EEG.info]=Load_EEG(fullfile(BMI.EEG_DIR, test),...
    {'device','brainVision';'marker',marker;'fs',opt.fs});
cnt=opt_eegStruct({EEG.data,EEG.marker,EEG.info},{'x','t','fs','y_dec','y_logic','y_class','class','chan'});

bufferSize=opt.fs*opt.bufferSize;
windowSize=opt.fs*opt.windowSize;
stepSize=opt.fs*opt.stepSize;

t=1:stepSize:length(cnt.x);
t(t+bufferSize>length(cnt.x))=[];
tt=t+bufferSize-windowSize;
LABEL=zeros(size(tt));

for m=1:length(cnt.t)
for k=1:length(tt)
    if cnt.t(m)<tt(k) && tt(k)<cnt.t(m)+opt.fs*duration
        LABEL(k)=cnt.y_dec(m);
    end
end
end
