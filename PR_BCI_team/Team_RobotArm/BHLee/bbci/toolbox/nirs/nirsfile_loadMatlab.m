function varargout= nirsfile_loadMatlab(file, varargin)
% NIRSFILE_LOADMATLAB - Load NIRS data structure from Matlab file
%
% Synopsis:
%   [DAT, MRK, MNT,NFO...]= nirsfile_loadMatlab(FILE, VARS)
%   [DAT, MRK, MNT,...]= nirsfile_loadMatlab(FILE, 'Property1', Value1, ...)
%
% Arguments:
%   FILE: name of data file
%   VARS: Variables (cell array of strings) which are to be loaded,
%         default {'dat','mrk','mnt'}. The names 'dat', 'cnt' and 'epo'
%         are treated equally.
%
% Returns:
%   Variables in the order specified in VARS. Default [DAT,MRK,MNT,NFO] or
%   less, depending on the number of output arguments.
%
% Properties:
%   'path': Path to save the file. Default is the global variable NIRS_MAT_DIR
%           (if exists; otherwise EEG_MAT_DIR) unless FILE is an absolute path.
%   'vars': cell array of variables to-be-loaded (default 
%           {'dat','mrk','mnt','nfo'}. The order corresponds with the order
%           of the output arguments.
%   'clab': Channel labels (cell array of strings) for loading a subset of
%           all channels. Default 'ALL' means all available channels.
%           In case OPT.clab is not 'ALL' the electrode montage 'mnt' is 
%           adapted automatically.
%   'signal' : which signal should be contained in the cnt.x-field: 'oxy',
%             'deoxy' (default), 'oxy-deoxy' (or 'both').
%   'verbose' : 0 (default) or 1
%
% See also: nirsfile_*  nirs_*
% Note: Based on eegfile_loadMatlab.
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

if numel(varargin)==1 && iscell(varargin{1})
  opt= propertylist2struct('vars', varargin{1});
else
  opt= propertylist2struct(varargin{:});
end

datnames = {'dat','cnt','epo'};
default_vars= {'dat','mrk','mnt','nfo'};

[opt, isdefault]= ...
    set_defaults(opt, ...
                 'path', nirsdir, ...
                 'clab', 'ALL', ...
                 'vars', default_vars(1:min(4,nargout)), ...
                 'signal','deoxy' ...
                );

if ~iscell(opt.vars)
  opt.vars= {opt.vars};
end

if nargout~=length(opt.vars)
  warning('number of output arguments does not match with requested vars');
end

fullname= fullfile(opt.path,file);
iData= find(ismember(opt.vars, datnames));

%% Load non-data variables
load_vars= setdiff(opt.vars,opt.vars{iData});
S= load(fullname, load_vars{:});

%% Check whether all requested variables have been loaded.
missing= setdiff(load_vars, fieldnames(S));
if ~isempty(missing)
  error(['Variables not found: ' sprintf('%s ',missing{:})]);
end

%% Adapt electrode montage, if only a subset of channels is requested
if isfield(S, 'mnt') && ~isequal(opt.clab, 'ALL'),
  S.mnt= mnt_adaptMontage(S.mnt, opt.clab);
end

%% Load data
if ~isempty(iData)
  names = whos('-file',fullname);  % Check name of dat file
  names = {names.name};
  name = intersect(datnames,names);
  if isempty(name)
    error(['Neither of the data variables found: ' sprintf('%s ',datnames{:})])
  elseif numel(name)>1
    warning(sprintf('Found multiple data variables (%s), taking ''%s''\n',sprintf('''%s'' ',name{:}),name{1}))
    name = name{1};
  else
    name = name{:};
  end
  dat = load(fullname,name);
  dat = dat.(name);
  switch(opt.signal)
    case 'oxy'
        dat.x = dat.x(:,1:end/2);
        dat.clab = dat.clab(:,1:end/2);
    case 'deoxy' 
        dat.x = dat.x(:,end/2+1:end);
        dat.clab = dat.clab(:,end/2+1:end);
    case {'oxy-deoxy' 'both'}
        % nix
    otherwise error('Unknown signal %s',opt.signal)
  end
end

dat.xInfo = opt.signal;

%% Output arguments
for vv= 1:nargout
  if ismember(vv, iData),
    varargout(vv)= {dat};
  else
    varargout(vv)= {getfield(S, opt.vars{vv})};
  end
end
