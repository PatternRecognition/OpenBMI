function [cnt,peak] = proc_pulse_detect(cnt,time,ma,varargin);

if length(cnt.clab)>1
  error('only one channel');
end

global BCI_DIR

path(path,[BCI_DIR 'fmrib1.2b/']);

eeg = struct('srate',cnt.fs,'data',cnt.x');

peak = fmrib_qrsdetect(eeg,1);

ind = find(diff(peak)<time*cnt.fs/1000);

peak(ind+1)=[];

pulse = diff(peak/cnt.fs);
pulse = 60./pulse;

cnt.x(1:peak(2)-1)= pulse(1);
for i = 2:length(peak)-1
  cnt.x(peak(i):peak(i+1)-1) = pulse(i);
end
cnt.x(peak(end):end) = pulse(end);


cnt = proc_movingAverage(cnt,ma,varargin{:});

