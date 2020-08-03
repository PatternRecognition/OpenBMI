% length in sec

function playWhiteNoise(len,fs)

t = [0:(1/fs):len-(1/fs)];
len_t = length(t);

whi = randn(2,len_t);

wavplay(whi',fs)