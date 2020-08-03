function dat= proc_filtNotchbyFFT(dat, Notch, N, L, windowfunction)
%dat= proc_filtByFFT(dat, band, <N, L=N>)
%
% dat  - data structure
% Notch : frequency to eliminate in Hz
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
% warning: please take care of the fact that Notch has to be a
% divisor of N.

% by guido

if ~exist('Notch','var'), Notch =50;end
if ~exist('N','var'), N= dat.fs; end
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


[T, nCE]= size(dat.x);


bInd = getNotchIndices(Notch,dat.fs,N);
bInd= [bInd,N+2-bInd(find(bInd>1))];
specMask = ones(N,1);
specMask(bInd) = 0;

xf= zeros(size(dat.x));
outWin= 1:N;
while outWin(end)<=T,
  Y= fft(feval(windowfunction,dat.x(outWin,:),arguments), N);
  Y= timesMatVec(Y, specMask);
  xf(outWin,:)= xf(outWin,:) + real(ifft(Y));
  outWin= outWin+L;
end
dat.x= xf;
