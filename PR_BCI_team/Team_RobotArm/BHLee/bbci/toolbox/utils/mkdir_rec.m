function s = mkdir_rec(path)
%
% s = mkdir_rec(path)
%
% path    directory to be created
% s       1 on success
%
% Creates a new directory. Contrary to mkdir, also creates the complete
% path if it does not exists (e.g. assume /home/john already exists. Then
% mkdir_rec('/home/john/test_1/test_2/test 3') would create 'test_1',
% 'test_2', and 'test 3'. Works with relative and absolute pathes.
%  
% $Id: mkdir_rec.m,v 1.4 2007/02/06 10:50:23 neuro_toolbox Exp $
%
% Copyright (C) 2006 Fraunhofer FIRST
% Author: Konrad Rieck (rieck@first.fraunhofer.de)

 
s = 1;
if isunix
    status = unix(sprintf('bash -c "mkdir -p %s"', path));
else
    % the DOS version results in an error if any slashes occur.
    path(find(ismember(path,'/')))='\';
    % The following works recursively if Command Extensions are enabled:
    status = system(['mkdir ' path]);
end
if (status ~= 0)
    error(sprintf('Could not create directory "%s".', path));
    s = 0;
end

return

% Alter Code:
%
% Copyright (C) 2001 Fraunhofer FIRST
% Author: Sebastian Mika (mika@first.fraunhofer.de)
%
% Ach, Sebastian, ist das kompliziert. 

  p = pathparts(path);
  
  num = size(p,2);

  oldd = pwd;
  
  if (isempty(oldd))
    oldd = '/';
  end
  
  for i=1:num
    if (unix(sprintf('bash -c "cd %s >& /dev/null"', p{i})) ~= 0)
      status = unix(sprintf('bash -c "mkdir %s"', p{i}));
      if (status ~= 0)
	cd(oldd);
	error(sprintf('Could not create part "%s" of "%s".', p{i}, path));
      else
	cd(p{i});
      end
    else
      cd(p{i});
    end
  end
  
  cd(oldd);

  s = 1;
  return
