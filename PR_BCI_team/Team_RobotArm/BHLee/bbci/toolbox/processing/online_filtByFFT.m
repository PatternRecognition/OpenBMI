function [dat,state]= online_filtByFFT(dat, state,band, N, L, windowfunction)
%dat= online_filtByFFT(dat, state,band, <N, L=N>)
%
% dat  - data structure
% band : [lower upper] in Hz
% N    : length of FFT
% L    : step of moving window (L<N: overlapping)
% windowfunction: a function which change the window (range
% 0:1). default (f(x,values) = sin^2(pi*x)*values) 
% here are some possibilities:
% first a function and optional given arguments for the call of the
% function (then in a cell array, only second to last argument), 
% x must be the first argument and says this is the place for the datas
% second a function and optional given arguments for the call of
% the function (then in a cell array (all arguments)), 
% x is not an argument, only a  factor then will be calculated. 
% After this factor*x is calculated.
% nothing. then f(x,N) = sin(pi*(1:N)/N).^2.*x;
%
% warning: last part may be left zero.

% by guido

if ~exist('N','var'), N= 2^nextpow2(dat.fs); end
if ~exist('L','var'), L= N/2; end
if ~exist('windowfunction') | isempty(windowfunction) 
  windowfunction = {inline('sin(pi*(1:N)/N).^2','N'), N};
end
if iscell(windowfunction)
  arguments  = cat(1,windowfunction{2:end});
  windowfunction = windowfunction{1};
else
  arguments = [];
end
if ~iscell(arguments)
  arguments = {arguments};
end


xfound = sum(strcmp(symvar(windowfunction),'x'))>0;

if xfound == 0
  arguments = feval(windowfunction,arguments{:});
  windowfunction = inline('(transpose(a)*ones(1,size(x,2))).*x','x','a'); 
end



bInd= getBandIndices(band, dat.fs, N);
bNeg= N+2-bInd(find(bInd>1));
specMask= zeros(N, 1);
specMask([bInd bNeg])= 1;

state = cat(1,state,dat.x);

[T, nCE]= size(state);

if T<N
  return
end


xf= zeros(T,nCE);


outWin= (1:N)+mod(T-N,L);
while outWin(end)<=T,
  Y= fft(feval(windowfunction,state(outWin,:),arguments), N);
  Y= timesMatVec(Y, specMask);
  xf(outWin,:)= xf(outWin,:) + real(ifft(Y));
  outWin= outWin+L;
end
dat.x= xf(end-size(dat.x,1)+1:end,:);
state = state(end-N+1:end,:);


