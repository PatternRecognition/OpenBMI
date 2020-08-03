%function htlt
%HTLT 	Unit test for the function htl.
 
%	O. Lemoine - December 1995. 

% Test for one slant line in a binary image (rho=20,theta=pi/4)

M=64; N=64;			% Size of the image IM
rho=20; theta=pi/4;		% Position of the line
IM=lineplot(rho,theta,M,N);	% Binary image IM
Mr=128; Nt=128;			% Resolution along rho and theta
[HT,r,t]=htl(IM,Mr,Nt);		% Hough transform
[Max,tmax]=max(max(HT));
[Max,rmax]=max(max(HT'));

if abs(r(rmax)-rho)>10/Mr,	% Position of the max along the rho
  error('htl test 1 failed');	% axis
elseif abs(t(tmax)-theta)>10/Nt, % Position of the max along the theta
  error('htl test 2 failed');	% axis
elseif length(find(HT>Max/5))>1, % Test if only one peak
  error('htl test 3 failed');	%  
end

% Test for foor lines (rho=[10 10 10 10],theta=[0 pi/2 pi 3*pi/2])

M=64; N=64;			% Size of the image IM
rho=[10 10 10 10]; 
theta=[0 pi/2 pi 3*pi/2];	
IM=lineplot(rho,theta,M,N);	% Binary image IM
Mr=128; Nt=128;			% Resolution along rho and theta
[HT,r,t]=htl(IM,Mr,Nt);		% Hough transform

[Max,rmax]=max(max(HT'));

if abs(r(rmax)-rho(1))>10/Mr,	% Position of the max along the rho
  error('htl test 4 failed');	% axis
end
for k=1:4,
  th=find(abs(t-theta(k))<.01);
  Maxi(k)=HT(rmax,th);
  if length(find(HT>Max/5))>4,  % Test if only foor peaks
    error('htl test 5 failed');	%  
  end
  if abs(Maxi(k)-Max)>1000*eps,
    error('htl test 6 failed');	% Test if same foor peaks
  end
end



% Test for one slant line in a binary image (rho=20,theta=pi/4)

M=63; N=63;			% Size of the image IM
rho=20; theta=pi/4;		% Position of the line
IM=lineplot(rho,theta,M,N);	% Binary image IM
Mr=127; Nt=127;			% Resolution along rho and theta
[HT,r,t]=htl(IM,Mr,Nt);		% Hough transform
[Max,tmax]=max(max(HT));
[Max,rmax]=max(max(HT'));

if abs(r(rmax)-rho)>100/Mr,	% Position of the max along the rho
  error('htl test 7 failed');	% axis
elseif abs(t(tmax)-theta)>10/Nt, % Position of the max along the theta
  error('htl test 8 failed');	% axis
elseif length(find(HT>Max/5))>1, % Test if only one peak
  error('htl test 9 failed');	%  
end
