%% Band-Pass filter visualization

[b, a] = butter(4, [0.5 50] / 100, 'bandpass');
y = filter(b, a, cnt.x);
cnt.x=cnt.x(:,1:34);
cnt.clab=cnt.clab(:,1:34);

f_sz=ceil(length(cnt.x)/2);
f=100*linspace(0,1,f_sz);
f_X=fft(cnt.x);
f_y=fft(y);
subplot(2,1,1)
stem(f,abs(f_X(1:f_sz)));
title('Original signal');
xlabel('frequency');
xlim([0 50]);
ylabel('power');
subplot(2,1,2)
stem(f,abs(f_y(1:f_sz)));
xlim([0 50]);

title('Application of the beta (13-30Hz) bandpass filter')
xlabel('frequency');
ylabel('power');