function [x,y,z]= abr2xyz(a,b,r)
%[x,y,z]= abr2xyz(a,b,r)

if nargin<3, r=1; end

a= a*pi/180;
b= b*pi/180;

x= r .* sin(a) .* cos(b);
y= r .* sin(b);
z= r .* cos(a) .* cos(b);
