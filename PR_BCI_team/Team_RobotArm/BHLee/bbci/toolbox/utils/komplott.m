function h = komplott(x,t)
%KOMPLOTT plots the rows of x non-overlapping into one axis.
%using time axis t

% STH * 22JAN2002 
%modified by AZ ; Nov 2005

if ndims(x) ~= 2,
  error('x must be a matrix')
end
[rx,cx] = size(x);
if rx>cx
  x = x';
  [rx,cx] = size(x);
end


if ~exist('t','var'),
  t=1:cx;
end

  
ron = ones(1,cx);  % a row of ones

% shift the signals such that the means along the rows are zero
x = x - mean(x,2)*ron;

% rescale the signals such that the absolute maximum is one
x = x ./ (2*max(abs(x),[],2)*ron);

% shift the signals such that they are distributed over the axis
% note, the -1 is necessary because we reverse the y axis later on
x = (-1)*x + (1:rx)'*ron;

hh = plot(t,x');
set(gca,'YDir','reverse')
set(gca,'YTick',1:rx)
space = 0.1;
set(gca,'YLim',[0.5-space rx+0.5+space])
set(gca,'XLim',[min(t) max(t)])
if nargout > 1
  h = hh;
end