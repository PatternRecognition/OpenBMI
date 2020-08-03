echo off
%PARAMFUN Figures representing different TFD of the Cohen's class.
%	On the left, the ambiguity plane and the weighting functions ;
%	On the right, the time-frequency distributions.

%	O. Lemoine - February, July 1996. 


load paramfun
Ncont=5;

subplot(321);
contour(dlr([(N+rem(N,2))/2+1:N 1:(N+rem(N,2))/2],:),8); 
xlabel('Delay'); ylabel('Doppler');
title('Wigner-Ville weighting function')
set(gca,'yticklabels',[])
set(gca,'xticklabels',[])
hold on
[a,h]=contour(WF1,[1/2],'g');
set(h,'linewidth',2);
hold off

subplot(322);
Max=max(max(tfr1));
levels=linspace(Max/10,Max,Ncont);
contour(tfr1,levels);
xlabel('Time'); ylabel('Frequency');
title('Wigner-Ville distribution')
set(gca,'yticklabels',[])
set(gca,'xticklabels',[])

subplot(323);
contour(dlr([(N+rem(N,2))/2+1:N 1:(N+rem(N,2))/2],:),8); 
xlabel('Delay'); ylabel('Doppler');
title('Spectrogram weighting function');
set(gca,'yticklabels',[])
set(gca,'xticklabels',[])
hold on
[a,h]=contour(WF2,[1/2],'g');
set(h,'linewidth',2);
hold off

subplot(324);
Max=max(max(tfr2));
levels=linspace(Max/10,Max,Ncont);
contour(tfr2(1:N/2,:),levels);
xlabel('Time'); ylabel('Frequency');
title('Spectrogram')
set(gca,'yticklabels',[])
set(gca,'xticklabels',[])

subplot(325);
contour(dlr([(N+rem(N,2))/2+1:N 1:(N+rem(N,2))/2],:),8); 
xlabel('Delay'); ylabel('Doppler');
title('SP-WV weighting function');
set(gca,'yticklabels',[])
set(gca,'xticklabels',[])
hold on
[a,h]=contour(WF3,[1/2],'g');
set(h,'linewidth',2);
hold off

subplot(326);
Max=max(max(tfr3));
levels=linspace(Max/10,Max,Ncont);
contour(tfr3,levels);
xlabel('Time'); ylabel('Frequency');
title('Smoothed-pseudo-WVD');
set(gca,'yticklabels',[])
set(gca,'xticklabels',[])


figure(2);

subplot(221);
contour(dlr([(N+rem(N,2))/2+1:N 1:(N+rem(N,2))/2],:),8); 
xlabel('Delay'); ylabel('Doppler');
title('Born-Jordan weighting function');
set(gca,'yticklabels',[])
set(gca,'xticklabels',[])
hold on
[a,h]=contour(WF4,[1/2],'g');
set(h,'linewidth',2);
hold off

subplot(222);
Max=max(max(tfr4));
levels=linspace(Max/10,Max,Ncont);
contour(tfr4,levels);
xlabel('Time'); ylabel('Frequency');
title('Born-Jordan distribution');
set(gca,'yticklabels',[])
set(gca,'xticklabels',[])


subplot(223);
contour(dlr([(N+rem(N,2))/2+1:N 1:(N+rem(N,2))/2],:),8); 
xlabel('Delay'); ylabel('Doppler');
title('CW weighting function');
set(gca,'yticklabels',[])
set(gca,'xticklabels',[])
hold on
[a,h]=contour(WF5,[1/2],'g');
set(h,'linewidth',2);
hold off

subplot(224);
Max=max(max(tfr5));
levels=linspace(Max/10,Max,Ncont);
contour(tfr5,levels);
xlabel('Time'); ylabel('Frequency');
title('Choi-Williams distribution');
set(gca,'yticklabels',[])
set(gca,'xticklabels',[])

echo on