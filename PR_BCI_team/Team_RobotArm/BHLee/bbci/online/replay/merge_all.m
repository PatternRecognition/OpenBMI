function merge_all(opt,file,varargin);


str = sprintf('%s ',varargin{:});
[fi,di] = separate_dir_file(file);

di2 = pwd;
cd(di);
str = sprintf('avimerge -o %s -i %s',fi,str);

system(str);


cd(di2);
