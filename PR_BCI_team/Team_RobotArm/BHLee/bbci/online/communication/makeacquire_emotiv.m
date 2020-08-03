function makeacquire_emotiv

try
    acquire_emotiv('close');
end
clear functions

params = {'emotiv\edk.lib emotiv\edk_utils.lib'};
params = ['acquire_emotiv.c' params];
params = ['-compatibleArrayDims' '-v' '-O' params];

mex -v acquire_emotiv.c emotiv\edk.lib emotiv\edk_utils.lib 
disp('Build completed.')
