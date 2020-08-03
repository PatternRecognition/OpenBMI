function [m]=get_rand_double_order(m)

[m,order]=get_random_order(m);
no_rows=size(m,1);
no_columns=size(m,2);

m=[m;m];
m=m(:);
m=reshape(m,no_rows,no_columns*2);




