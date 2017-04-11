function acc=ssvep_acc(smt,varargin)
% Example:
%     acc=ssvep_acc(smt,{'time',[0:0.01:5];'frequency',[7.5 10 12];'channel',{'Oz','O1','O2'}});

opt=opt_cellToStruct(varargin{:});
if ~isfield(opt,'time')
    error('OpenBMI: No time information')
elseif ~isfield(opt,'frequency')
    error('OpenBMI: No frequency information')
elseif ~isfield(opt,'channel')
    error('OpenBMI: No channel information')
elseif isnumeric(opt.channel)
    channel=opt.channel;
elseif ischar(opt.channel) || iscell(opt.channel)
    channel=find(ismember(smt.chan,opt.channel));
end
[~,nTrial,~]=size(smt.x);
%% adjusting CCA algorithm
t=opt.time; f=opt.frequency;
for i=1:length(f)
    Y{i}=[sin(2*pi*f(i)*t);cos(2*pi*f(i)*t);sin(2*pi*2*f(i)*t);cos(2*pi*2*f(i)*t);];
end

for i=1:nTrial
    dat.x=smt.x(:,i,channel);
    
    dat.x=squeeze(dat.x);
    dat.fs=100;
    dat=prep_filter(dat,{'frequency', [5 40]});
    for j=1:length(Y)
        [~,~,cor_r{j}]=canoncorr(dat.x,Y{j}');
    end
    r(i,:)=cellfun(@mean,cor_r);
    [~,j]=max(r(i,:));
    out(i)=j;
end

[a,~]=find(smt.y_logic==1);
[a2,~]=find(out'==a);
acc=(length(a2)/length(out))*100;

