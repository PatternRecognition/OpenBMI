function racing_plot(clfLabel,trueLabel,varargin)

opt=opt_cellToStruct(varargin{:});
if isfield(opt,'time')
    t=opt.time;
else
    t=1:length(clfLabel);
end
figure()
stem(t,clfLabel,'r')
hold on

c={'m','c','y','k'};
% shadowing MI range
u=unique(trueLabel);
for i=1:length(u)-1
    idx1=find(trueLabel==i);
    idx2=[1,find(diff(idx1)~=1)+1];
    for j=1:length(idx2)-1
    p=patch([t(idx1(idx2(j))),t(idx1(idx2(j+1)-1)),t(idx1(idx2(j+1)-1)),t(idx1(idx2(j)))],[-10 -10 10 10],c{i});
    set(p,'FaceAlpha',0.3);
    end
    j=length(idx2);
    p=patch([t(idx1(idx2(j))),t(idx1(end)),t(idx1(end)),t(idx1(idx2(j)))],[-1 -1 6 6],c{i});
    set(p,'FaceAlpha',0.3);
end
ylim([-1 6])

ox=clfLabel==trueLabel;
idx=find(ox);
% scatter(t(idx),ox(idx),'*')
stem(t(idx),clfLabel(idx),'b')
