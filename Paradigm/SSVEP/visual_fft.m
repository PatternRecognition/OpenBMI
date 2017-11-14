function visual_fft(smt,varargin)
% Example:
%     visual_fft(smt,{'channel','Oz';'xlim',[5 20]})

opt=opt_cellToStruct(varargin{:});
if ~isfield(opt,'channel')
    error('OpenBMI: No channel information')
    % elseif numel(opt.channel)~=1
    %     error('OpenBMI: Enter only one channel')
elseif isnumeric(opt.channel)
    channel=opt.channel;
elseif ischar(opt.channel) || iscell(opt.channel)
    channel=find(ismember(smt.chan,opt.channel));
end

if ~isfield(opt,'plot')
    opt.plot='off';
end

avg=prep_average(smt);
avg=prep_selectChannels(avg,{'Index',channel});

f=figure;
for i=1:size(avg.class,1)
    [YfreqDomain(i,:),frequencyRange] = positiveFFT(avg.x(:,i),avg.fs);
%     YY(:,i)=mag2db(abs(YfreqDomain));
    if strcmp(opt.plot,'on')
        plot(frequencyRange,abs(YfreqDomain(i,:))); hold on;
    end
end
total = size(smt.x,2);
loss = total;
for i=1:total
   [~, ind]=max(ssvep_cca_analysis(squeeze(smt.x(:,i,:)),{'marker',smt.class;'freq', [5 7 9 11];'fs', smt.fs;'time',4}));
   if ~isequal(ind, smt.y_dec(i)) 
       loss = loss - 1; 
   end
end
Accuracy =loss/total;

% y_line=[0 ceil(max([max(abs(YfreqDomain))]))];
y_line= get(gca,'Ylim');

if strcmp(opt.plot,'on')
    
    xlabel('Frequency [Hz]'),ylabel('Power')
    
    % draw line
    if isfield(opt, 'line') && opt.line
        line([60/5 60/5], y_line, 'Color', 'k'); hold on;
        line([60/7 60/7], y_line, 'Color', 'k'); hold on;
        line([60/9 60/9], y_line, 'Color', 'k'); hold on;
        line([60/11 60/11], y_line, 'Color', 'k'); hold on;
    end
    if isfield(opt,'xlim')
        xlim(opt.xlim)
    end
    if isfield(opt,'title')
        title(opt.title)
    end
    c_num=length(smt.class(:,2)); % legend having the number of classes
    i_legend=cell(c_num,1);
    for i=1:c_num
        i_legend{i}=char(smt.class(i,2));
    end
    legend(i_legend);
end
title(sprintf('%s / %s / %s / acc: %.2f%%',opt.subject_info.subject,opt.subject_info.session,opt.filename, Accuracy*100),'Interpreter','none');
saveas(f,sprintf('%s\\figure\\%s_%s_%s.jpg',opt.filepath,opt.subject_info.subject,opt.subject_info.session,opt.filename));
end

function [X,freq]=positiveFFT(x,Fs)
N=length(x);    % get the number of points
k=0:N-1;        % create a vector from 0 to N-1
T=N/Fs;         % get the frequency interval
freq=k/T;       % create the frequency range
X=fft(x)/N*2;   % normalize the data
cutOff = ceil(N/2);

% take only the first half of the spectrum
X = X(1:cutOff);
freq = freq(1:cutOff);
end