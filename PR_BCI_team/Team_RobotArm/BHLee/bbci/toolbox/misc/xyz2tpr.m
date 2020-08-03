function [t,p,r]= xyz2tpr(x,y,z)

r= sqrt(x.^2 + y.^2 + z.^2);
p= atan(y./x);
xy= sqrt(x.^2 + y.^2);
t= atan(xy./z);

t= abs(t*180/pi);
negx= find(x<0);
t(negx)= -t(negx);
p= p*180/pi;
p(isnan(p))= 0;
