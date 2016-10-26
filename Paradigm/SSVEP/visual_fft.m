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

avg=prep_average(smt);
avg=prep_selectChannels(avg,{'Index',channel});
figure()
for i=1:size(avg.class,1)
    [YfreqDomain,frequencyRange] = positiveFFT(avg.x(:,i),avg.fs);
    plot(frequencyRange,abs(YfreqDomain)); hold on;
end
if isfield(opt,'xlim')
    xlim(opt.xlim)
end
xlabel('Frequency[Hz]'),ylabel('Amplitude[mV]')
legend(str2mat(smt.class(1,2)),str2mat(smt.class(2,2)),str2mat(smt.class(3,2)))

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