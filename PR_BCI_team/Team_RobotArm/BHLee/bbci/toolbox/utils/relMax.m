function y= relMaxs(x)

y= find(diff(sign(diff(x)))<-1)+1;
