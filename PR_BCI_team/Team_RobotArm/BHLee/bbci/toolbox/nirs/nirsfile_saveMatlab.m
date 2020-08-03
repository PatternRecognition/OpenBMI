function nirsfile_saveMatlab(file, dat, mrk, mnt, varargin)
% NIRSFILE_SAVEMATLAB - Save NIRS data structures in Matlab format
%
% Synopsis:
%   nirsfile_saveMatlab(FILE, DAT, MRK, MNT, 'Property1', Value1, ...)
%
% Arguments:
%   FILE: name of data file
%   DAT: structure of continuous or epoched signals
%   MRK: marker structure
%   MNT: electrode montage structure
%
% Properties:
%   'path': Path to save the file. Default is the global variable EEG_MAT_DIR
%           unless FILE is an absolute path.
%   'vars': Additional variables that should be stored. 'opt.vars' must be a
%           cell array with a variable name / variable value strucutre, e.g.,
%           {'hdr',hdr, 'blah',blah} when blub and blah are the variables
%           to be stored.
%   'fs_orig': store information about the original sampling rate
%
% See also: nirsfile_*  nirs_*
% Note: Based on eegfile_saveMatlab
%
% matthias.treder@tu-berlin.de 2011

if ~isempty(whos('global','NIRS_MAT_DIR'))
  global NIRS_MAT_DIR
  nirsdir = NIRS_MAT_DIR;
else
  global EEG_MAT_DIR
  nirsdir = EEG_MAT_DIR;
end

if isabsolutepath(file)
  nirsdir = [];
end

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'path', nirsdir, ...
                 'fs_orig', [], ...
                 'vars', {});

if ~iscell(opt.vars), opt.vars = {opt.vars}; end

fullname= fullfile(opt.path,file);
dat.file= fullname;

%% Gather some summary information into structure 'nfo'.
nfo= copy_struct(dat, 'fs', 'clab','source','detector');
if isfield(dat,'x')
  nfo.T= size(dat.x,1);
end
if isfield(dat,'x') && ndims(dat.x)>2
  nfo.nEpochs= size(dat.x,3);
else
  nfo.nEpochs = 1;
end
nfo.length= nfo.T * nfo.nEpochs / dat.fs; % Length in seconds
nfo.file= fullname;
if isfield(mrk, 'pos'),
  nfo.nEvents= length(mrk.pos);
else
  nfo.nEvents= 0;
end
if isfield(mrk, 'y'),
  nfo.nClasses= size(mrk.y,1);
else
  nfo.nClasses= 0;
end
if isfield(mrk, 'className'),
  nfo.className= mrk.className;
else
  nfo.className= {};
end
if ~isempty(opt.fs_orig),
  nfo.fs_orig= opt.fs_orig;
end

%% Create directory if necessary
[filepath, filename]= fileparts(fullname);
if ~exist(filepath, 'dir'),
  [parentdir, newdir]=fileparts(filepath);
  [status,msg]= mkdir(parentdir, newdir);
  if status~=1,
    error(msg);
  end
  if isunix,
    unix(sprintf('chmod a-rwx,ug+rwx %s', filepath));
  end
end

save(fullname, 'dat', 'mrk', 'mnt', 'nfo');

%% Save additional variables, as requested.
if ~isempty(opt.vars)
  for vv= 1:length(opt.vars)/2,
    eval([opt.vars{2*vv-1} '= opt.vars{2*vv};']);
  end
  save(fullname, '-APPEND', opt.vars{1:2:end});
end
