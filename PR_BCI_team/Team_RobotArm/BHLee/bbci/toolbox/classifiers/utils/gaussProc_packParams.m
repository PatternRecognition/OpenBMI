function p = gaussProc_packParams(opt)
% gaussProc_packParams - GP helper: Convert model options to vector
%
% Synopsis:
%   p = gaussProc_packParams(opt)
%   

nKernels = length(opt.kernel);
N = 0;
% First compute the total length of the parameter vector (yes, I'm a bit
% picky about pre-allocating memory, instead of growing vectors)
for i = 1:nKernels,
  N = N+length(opt.kernel{i}{2}.allParams);
end
p = zeros([1 N+length(opt.allParams)]);
start = 1;
% Extract parameters for each kernel individually and pack into a long vector
for i = 1:nKernels,
  kernelOpt = opt.kernel{i}{2};
  Ni = length(kernelOpt.allParams);
  p(start:(start+Ni-1)) = struct2vect(kernelOpt, kernelOpt.allParams);
  start = start+Ni;
end
p(start:end) = struct2vect(opt, opt.allParams);
