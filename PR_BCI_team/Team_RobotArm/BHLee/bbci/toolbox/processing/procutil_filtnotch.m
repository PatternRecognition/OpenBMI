function [B, A]= procutil_filtnotch(fa, fg, r)

if ~exist('r','var')  | isempty(r),  r=0.98; end

w=2*pi*fg/fa;

%Coefficient r

if r>0 & r<1
  
  
  %r defines the damping and band
elseif -50<r & r<-17
  for i=1:99
    n=i*0.01;
    B=[1 -2*cos(w) 1];
    A=[1 -2*n*cos(w) n^2];
    [H,F]=freqz(B,A,512,fa);
    d(i)=min(20*log10(abs(H)));
    x(i)= n;   
  end  
  [P,S]=polyfit(x,d,6);
  r= fzero(@f,0.5,[],P,r);
  
  %r defines the distance between the Cut-Off-Frequencies
elseif r>1 & r<1522
  lin=0;
  rec=0;
  for l=1:49
    n(l)=0.02*l;
    B=[1 -2*cos(w) 1];
    A=[1 -2*n(l)*cos(w) n(l)^2];
    [H,F]=freqz(B,A,[],fa);
    for i=1:10
      lin=lin+20*log10(abs(H(i)));
      rec=rec+20*log10(abs(H(size(H,1)-i+1)));
    end
    lin=lin/10;
    rec=rec/10;
    j=1;
    while (20*log10(abs(H(j)))>(lin-3))
      j=j+1;
    end
    k=size(H,1);
    while (20*log10(abs(H(k)))>(rec-3))
      k=k-1;
    end  
    diff(l)=k-j;
  end
  [P,S]=polyfit(n,diff,2);
  r=r*2*512/fa;
  r=fzero(@fk,0.5,[],P,r);
  
  
else
  error('Please retry to design the notch filter with another r')
end

%Filtercoefficents
B=[1 -2*cos(w) 1];
A=[1 -2*r*cos(w) r^2];




function y=f(x,P,g)
y= P(1)*x^6 + P(2)*x^5 + P(3)*x^4 + P(4)*x^3 + P(5)*x^2 + P(6)*x^1 + P(7)- g;

function y=fk(x,P,g)
y= P(1)*x^2 + P(2)*x + P(3) -g;
