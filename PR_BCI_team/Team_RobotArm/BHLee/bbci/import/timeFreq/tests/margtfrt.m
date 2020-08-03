%function margtfrt
%MARGTFRT Unit test for the function MARGTFR.

%	O. Lemoine - April 1996.

N=128;

% WVD
sig=amgauss(N).*fmlin(N);
tfr=tfrwv(sig);
[margt,margf,E]=margtfr(tfr);
ip=abs(sig).^2;
if any(abs(ip-margt)>sqrt(eps)),
 error('margtfr test 1 failed');
end
psd=mean(tfr(1:2:N,:)')';
if any(abs(margf(1:2:N)-psd)>sqrt(eps)),
 error('margtfr test 2 failed');
end
Es=norm(sig)^2;
if any(abs(Es-E)>sqrt(eps)),
 error('margtfr test 3 failed');
end

% WVD ; 1 point / 2 in time
tfr=tfrwv(sig,1:2:128);
[margt,margf,E]=margtfr(tfr);
ip=abs(sig(1:2:128)).^2;
if any(abs(ip-margt)>sqrt(eps)),
 error('margtfr test 4 failed');
end
psd=mean(tfr(1:2:N,:)')';
if any(abs(margf(1:2:N)-psd)>sqrt(eps)),
 error('margtfr test 5 failed');
end
Es=norm(sig(1:2:128))^2;
if any(abs(Es-E)>sqrt(eps)),
 error('margtfr test 6 failed');
end

% WVD ; 1 point / 2 in frequency
tfr=tfrwv(sig,1:128,64);
[margt,margf,E]=margtfr(tfr);
ip=abs(sig).^2;
if any(abs(ip-margt)>sqrt(eps)),
 error('margtfr test 7 failed');
end
psd=mean(tfr(1:2:N/2,:)')';
if any(abs(margf(1:2:N/2)-psd)>sqrt(eps)),
 error('margtfr test 8 failed');
end
Es=norm(sig)^2;
if any(abs(Es-E)>sqrt(eps)),
 error('margtfr test 9 failed');
end

% Rihaczek distribution
tfr=tfrri(sig);
[margt,margf,E]=margtfr(tfr);
ip=abs(sig).^2;
if any(abs(ip-margt)>sqrt(eps)),
 error('margtfr test 10 failed');
end
psd=mean(tfr(1:2:N,:)')';
if any(abs(margf(1:2:N)-psd)>sqrt(eps)),
 error('margtfr test 11 failed');
end
Es=norm(sig)^2;
if any(abs(Es-E)>sqrt(eps)),
 error('margtfr test 12 failed');
end


N=117;

% WVD
sig=amgauss(N).*fmlin(N);
tfr=tfrwv(sig);
[margt,margf,E]=margtfr(tfr);
ip=abs(sig).^2;
if any(abs(ip-margt)>sqrt(eps)),
 error('margtfr test 13 failed');
end
psd=mean(tfr(1:2:N,:)')';
if any(abs(margf(1:2:N)-psd)>sqrt(eps)),
 error('margtfr test 14 failed');
end
Es=norm(sig)^2;
if any(abs(Es-E)>sqrt(eps)),
 error('margtfr test 15 failed');
end

% WVD ; 1 point / 2 in time
tfr=tfrwv(sig,1:2:N);
[margt,margf,E]=margtfr(tfr);
ip=abs(sig(1:2:N)).^2;
if any(abs(ip-margt)>sqrt(eps)),
 error('margtfr test 16 failed');
end
psd=mean(tfr(1:2:N,:)')';
if any(abs(margf(1:2:N)-psd)>sqrt(eps)),
 error('margtfr test 17 failed');
end
Es=norm(sig(1:2:N))^2;
if any(abs(Es-E)>sqrt(eps)),
 error('margtfr test 18 failed');
end

% WVD ; 1 point / 2 in frequency
tfr=tfrwv(sig,1:N,round(N/2));
[margt,margf,E]=margtfr(tfr);
ip=abs(sig).^2;
if any(abs(ip-margt)>sqrt(eps)),
 error('margtfr test 19 failed');
end
psd=mean(tfr(1:2:round(N/2),:)')';
if any(abs(margf(1:2:round(N/2))-psd)>sqrt(eps)),
 error('margtfr test 20 failed');
end
Es=norm(sig)^2;
if any(abs(Es-E)>sqrt(eps)),
 error('margtfr test 21 failed');
end

% Rihaczek distribution
tfr=tfrri(sig);
[margt,margf,E]=margtfr(tfr);
ip=abs(sig).^2;
if any(abs(ip-margt)>sqrt(eps)),
 error('margtfr test 22 failed');
end
psd=mean(tfr(1:2:N,:)')';
if any(abs(margf(1:2:N)-psd)>sqrt(eps)),
 error('margtfr test 23 failed');
end
Es=norm(sig)^2;
if any(abs(Es-E)>sqrt(eps)),
 error('margtfr test 24 failed');
end

