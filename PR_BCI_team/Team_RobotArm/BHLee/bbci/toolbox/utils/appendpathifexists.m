function f = appendpathifexists(pp, varargin)
%Appends a folder to the current path, if it exists.
%
%Returns:
% f - 1 if path was found, otherwise

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, 'verbose', 1, ...
                       'recursive', 1, ...
                       'exclude', '');

if exist(pp, 'dir')
  f = 1;
  if opt.recursive,
    ppp= genpath(pp);
    if ~isempty(opt.exclude),
      ppp= path_exclude(ppp, opt.exclude);
    end
    path(path, ppp);
  else
    path(path, pp);
  end
else
  f = 0;
  if opt.verbose,
    fprintf('Path <%s> not found: not added to Matlab''s search path.\n', pp);
  end
end
