
epo = epo2;
fs = epo.fs;
Y_t=[];
for j=1:size(epo.x,3)
    
    x = epo.x(:,:,j);
    T=1/fs;
    L=size(x,1);
    t=(0:L-1)*T;
    
    NFFT = 2^nextpow2(L); % Next power of 2 from length of y
    f = fs/2*linspace(0,1,NFFT/2);
    
    Y = fft(x,NFFT)/L;
    Y_t(:,:,j) = 2*abs(Y(1:NFFT/2,:));
    
end
Y_fft = mean(Y_t,3);
psd_Y = 10*log10(Y_fft);
figure; plot(f, Y_fft)