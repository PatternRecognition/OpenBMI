function makeacquire_bv(nouzz)

try
    acquire_bv('close');
end
clear functions

if isunix
    params = {'../winunix/winthreads.c' '../winunix/winevents.c' '-lrt'};
else
    params = {'wsock32.lib'};
end

params = ['acquire_bv.c' 'brainserver.c' 'headerfifo.c' params];

if nargin>0 && nouzz
    params = ['-output' 'acquire_nouzz' '-DBV_PORT=32163' params];
end

params = ['-compatibleArrayDims' '-v' '-O' params];

mex(params{:})
disp('Build completed.')
