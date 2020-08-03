function IM=lineplot(rho,theta,M,N)
%LINEPLOT Plot lines on a binary image.
%	IM=LINEPLOT(RHO,THETA,M,N) plots a binary image containing
%	lines parametrized by the polar vectors RHO and THETA.
%
%	Example :
%	 IM=lineplot(10,pi/3,64,64); pcolor(IM); 
%
%	See also HTL.
 
%	O. Lemoine - December 1995.

if length(rho)~=length(theta),
  error('RHO and THETA must have the same length');
end

N1=length(rho);
IM=zeros(N,M);

if rem(N,2)~=0,
  Xc=(N+1)/2; X0=1-Xc; Xf=Xc-1;
else
  Xc=N/2; X0=1-Xc; Xf=Xc;
end
if rem(M,2)~=0,
  Yc=(M+1)/2; Y0=1-Yc; Yf=Yc-1;
else
  Yc=M/2; Y0=1-Yc; Yf=Yc;
end

for k=1:N1,
  if abs(sin(theta(k)))<1000*eps,
    for y=Y0:Yf,
      x=round((rho(k)+y*sin(theta(k)))/cos(theta(k)));
      if (x>=X0)&(x<=Xf),
        IM(x+Xc,y+Yc)=1;
      end
    end   
  else
    for x=X0:Xf,
      y=round((-rho(k)+x*cos(theta(k)))/sin(theta(k)));
      if (y>=Y0)&(y<=Yf),
        IM(x+Xc,y+Yc)=1;
      end
    end   
  end
end
