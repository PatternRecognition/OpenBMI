function [b, a]= procutil_catfilters(varargin)

sos= zeros(0, 6);
g= [];

for ii= 1:length(varargin)/2,
  [sos0, g0]= tf2sos(varargin{ii*2-1}, varargin{ii*2});
  sos= cat(1, sos, sos0);
  g= cat(1, g, g0);
end
[b, a]= sos2tf(sos, g);
