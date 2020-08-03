function transfer_cvs_to_svn(varargin)
%TRNASFER_CVS_TO_SVN - Transfer old CVS Repository to SVN (update newer files)

% Authors: Konrad Grzeska, Benjamin Blankertz

global BCI_DIR

%warning('svn password has to be entered once before running this');

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'filelist', strcat(BCI_DIR, ...
                      'bbci_tools/include_transfer_bbcicvs.txt'), ...
                  'exclude', '', ...
                  'cvs_root', '~/neuro_cvs/matlab/bci', ...
                  'svn_root', '~/svn/ida/public/bbci', ...
                  'recursive', 1, ...
                  'test', 0);
%                  'svn_root', '~/svn/ida/public/bbci', ...

if ischar(opt.exclude) & exist(opt.exclude, 'file'),
  opt.exclude= textread(opt.exclude, '%s', ...
                        'commentstyle','shell');
end

fprintf('\n*** Processing filelist %s.\n', opt.filelist);
if opt.test,
  fprintf('*** TESTING MODE: Changes are only displayed.\n');
end

[source, target]= textread(opt.filelist, '%s %s', ...
                           'commentstyle','shell');

olddir= pwd;
memo_opt= opt;

for ii= 1:length(source)
  filespec= source{ii};
  if isempty(filespec),
    continue;
  end
  
  %% locally optional properties can be overridden, e.g.
  %% by prefixing a line with '[recursive=0]' in the transfer fileilst.
  opt= memo_opt;
  if filespec(1)=='[',
    ic= min(find(filespec==']'));
    optsetstr= filespec(2:ic-1);
    eval(['opt.' optsetstr]);
    filespec= deblank(filespec(ic+1:end));
  end
  
  fullfilespec= [opt.cvs_root '/' filespec];
  transfer_files(fullfilespec, target{ii}, ...
                 setfield(opt, 'init',ii==1));
end

%% Add directories and files to SVN and commit them
%% (not sure why this is done aposteriori).
if ~opt.test,
  cmd= sprintf('cd %s; svn add --force *', opt.svn_root);
  unix_cmd(cmd, 'could not add to svn');
  [dmy,user]= unix('whoami');
  msg= sprintf('auto transfer from cvs issued by %s', user(1:end-1));
  cmd= sprintf('cd %s; svn commit -m "%s" *', opt.svn_root, msg);
  unix_cmd(cmd, 'could not commit to svn');
end
cd(olddir);
return;




function transfer_files(fullfilespec, targetpath, opt)

persistent processed
if opt.init,
  processed= {};
end

while fullfilespec(end)=='/',
  fullfilespec(end)= [];
end

if exist(fullfilespec, 'dir'),
  sourcepath= fullfilespec;
  source= '*';
  [dmy, subdir]= fileparts(fullfilespec);
  targetpath= [targetpath '/' subdir];
else
  [sourcepath, source]= fileparts(fullfilespec);
end
sourcepathrel= strrep(sourcepath, [opt.svn_root '/'], '')

targetfullpath= [opt.svn_root '/' targetpath];

if ~exist(targetfullpath, 'dir'),
  sprintf('making new directory %s.\n', targetfullpath);
  if ~opt.test,
    mkdir_rec(targetfullpath);
%    unix_cmd(sprintf('svn add %s', targetfullpath), ...
%             'could not add new directory to svn');
  end
end

dd= dir(fullfilespec);
dd(strmatch('.', {dd.name}, 'exact'))= [];
dd(strmatch('..', {dd.name}, 'exact'))= [];

for j = 1:length(dd),

  filename= dd(j).name;
  if ~isempty(strpatternmatch(opt.exclude, [sourcepathrel '/' filename])),
    fprintf('excluding %s.\n', [sourcepath '/' filename]);
    continue;
  end
  if (dd(j).isdir),
    if opt.recursive && ~strcmpi(dd(j).name,'CVS'),
      fprintf('recursing into %s.\n', [sourcepath '/' dd(j).name]);
      transfer_files([sourcepath '/' dd(j).name '/*'], ...
                     [targetpath '/' dd(j).name], opt);
    else
      fprintf('skipping directory %s.\n', [sourcepath '/' dd(j).name]);
    end
    continue;
  end

  ip= max(find(filename=='.'));
  fileext= filename(ip+1:end);
  if filename(end)=='~' || strcmp(filename(1:2), '.#'),
    continue;
  end
  sourcefile= [sourcepath '/' filename];
  targetfile= [targetfullpath '/' filename];
  
  if ismember(sourcefile, processed),
    continue;
  end
  processed= cat(1, processed, {sourcefile});
  
  if exist(targetfile, 'file') && ~fileisnewer(sourcefile, targetfile),
    %fprintf('skipping file %s.\n', sourcefile);
    continue;
  end
%  fprintf('  %s', sourcefile);
  fprintf('  %s -->> %s\n', sourcefile, [targetpath '/' filename]);
  
  if ~opt.test,
    copyfile(sourcefile, targetfile);
%    unix_cmd(sprintf('svn add %s', targetfile));
  end
end
