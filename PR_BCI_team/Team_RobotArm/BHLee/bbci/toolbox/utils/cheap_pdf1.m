function [n, x, d]= cheap_pdf1(data, dec);

if nargin<2,
  dec= min(30, length(data));
end
stuetz= linspace(min(data), max(data), ceil(length(data)/dec));
if length(stuetz)==1,  % min(data)==max(data)
  n= 1;
  x= data;
  d= 1;
  return
end
n0= hist(data, stuetz);
n1= repmat(n0(:)', [dec 1]);
start= floor((numel(n1) - length(data))/2);
n= movingAverage(n1(start + [1:length(data)])', dec, 'centered');
x= linspace(min(data), max(data), length(data));

if nargout>2,
  d= interp1(x, n, data);
end
