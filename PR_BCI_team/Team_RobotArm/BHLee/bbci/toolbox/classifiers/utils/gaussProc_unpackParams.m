function opt = gaussProc_unpackParams(p, opt)
% gaussProc_unpackParams - GP helper: Copy parameters from vector to structure
%
% Synopsis:
%   opt = gaussProc_unpackParams(p,opt)
%   

start = 1;
% Extracting from vector to struct is a bit more tricky:
for i = 1:length(opt.kernel),
  kernelOpt = opt.kernel{i}{2};
  Ni = length(kernelOpt.allParams);
  opt.kernel{i}{2} = vect2struct(kernelOpt, p(start:(start+Ni-1)), ...
                                 kernelOpt.allParams);
  % We got all params for the i.th kernel, start again from the next
  % entry in the vector
  start = start+Ni;
end
% The last entries in the vector must correspond to the model parameters
opt = vect2struct(opt, p(start:end), opt.allParams);
