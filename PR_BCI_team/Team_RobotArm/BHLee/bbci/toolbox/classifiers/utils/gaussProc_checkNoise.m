function gaussProc_checkNoise(opt, N)
% gaussProc_checkNoise - Helper function: Check noise groups parameter
%
% Synopsis:
%   gaussProc_checkNoise(opt)

% Need to make some checks for consistency if we have examples with
% grouped noise
nNoise = prod(size(opt.noise));
if isempty(opt.noisegroups),
  nGroups = 1;
else
  nGroups = prod(size(opt.noisegroups));
end
if nNoise~=nGroups,
  error('Length of options ''noise'' and ''noisegroups'' must match');
end
groupError = 0;
for i = 1:length(opt.noisegroups),
  if islogical(opt.noisegroups{i}),
    if length(opt.noisegroups{i})~=N,
      groupError = 1;
    end
  else
    if any(opt.noisegroups{i}<1 | opt.noisegroups{i}>N),
      groupError = 1;
    end
  end
  if groupError,
    error(sprintf('Invalid index in option ''noisegroups'' at position %i', i));
  end
end
if length(opt.noisegroups)>1,
  if opt.verbosity>1,
    fprintf('Option ''noisegroups'': You are using %i individual noise variances.\n',...
            nGroups);
  end
end
