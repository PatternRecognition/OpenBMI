function [ time ] = func_getTimeMarker( eegFb, opt )
%FUNC_GETTIMEMAKER Summary of this function goes here
%   Detailed explanation goes here
if ~isfield(eegFb, 'x')
    error('Test data should be field of eeg.x');
end

[L CH]=size(eegFb.x);

w_size=str2num(opt.windowSize);
t=str2double(opt.windowSize)*(1/10000)*eegFb.hdr.fs; %time;
b_size=ceil(L/w_size);

buffer=cell(b_size,1);
s_=1;
for i=1:b_size
    if i==b_size
        buffer{i}=s_:L;
    else
    buffer{i}=s_:s_+w_size-1;
    s_=s_+w_size;
    end
end
time.buffer=buffer;
time.ite=b_size;
time.t=t;

end

