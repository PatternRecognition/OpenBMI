% mex -O pnet.c 'C:\Program Files\MATLAB\R2014a\sys\lcc64\lcc64\lib64\wsock32.lib' -DWIN32
 % mex -O pnet.c ws2_32.lib -DWIN32
% http://de.mathworks.com/support/compilers/R2011a/win64.html
% sdk, c++2010 express
% http://de.mathworks.com/support/compilers/R2011a/win64.html
H = bv_open('JohnKim-PC');
while 1
DATA = bv_read(H);
pause(0.1);
end
bv_close(H);
