function eegfile_saveMatlab(file, dat, mrk, mnt, varargin)
% EEGFILE_SAVEMATLAB - Save EEG data structures in Matlab format
%
% Synopsis:
%   eegfile_saveMatlab(FILE, DAT, MRK, MNT, 'Property1', Value1, ...)
%
% Arguments:
%   FILE: name of data file
%   DAT: structure of continuous or epoched signals
%   MRK: marker structure
%   MNT: electrode montage structure
%
% Properties:
%   'path': Path to save the file. Default is the global variable EEG_MAT_DIR
%           unless FILE is an absolute path in which case it is ''.
%   'channelwise': If true, signals are saved channelwise. This is an advantage
%                  for big files, because it allows to load selected channels.
%   'format': 'double', 'float', 'int16', or 'auto' (default).
%             In 'auto' mode, the function tries to find a lossless conversion
%             of the signals to INT16 (see property '.resolution_list'). 
%             If this is possible '.format' is set to 'INT16', otherwise it is
%             set to 'DOUBLE'.
%   'resolution': Resolution of signals, when saving in format INT16.
%                 (Signals are divided by this factor before saving.) The resolution
%                 maybe selected for each channel individually, or globally for all
%                 channels. In the 'auto' mode, the function tries to find for each
%                 channel a lossless conversion to INT16 (see property
%                 '.resolution_list'). For all other channels the resolution producing
%                 least information loss is chosen (under the resolutions that avoid
%                 clipping). Possible values:
%                 'auto' (default), 
%                 numerical scalar, or 
%                 numerical vector of length 'number of channels'
%                 (i.e., length(DAT.clab)).
%   'resolution_list': Vector of numerical values. These values are tested as
%                      resolutions to see whether lossless conversion to INT16 is
%                      possible. Default [1 0.5 0.1].
%   'accuracy': used to define "losslessness" (default 10e-10)
%   'add_channels': (true or false) Adds the channels in DAT to the
%                   existing MAT file.
%   'vars': Additional variables that should be stored. 'opt.vars' must be a
%           cell array with a variable name / variable value strucutre, e.g.,
%           {'blub',blub, 'blah',blah} when blub and blah are the variables
%           to be stored.
%   'fs_orig': store information about the original sampling rate
%
% See also: eegfile_*
%
% Author(s): Benjamin Blankertz, Feb 2005

global EEG_MAT_DIR

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'path', EEG_MAT_DIR, ...
                 'channelwise', 1, ...
                 'format', 'auto', ...
                 'resolution', 'auto', ...
                 'resolution_list', [1 0.5 0.1], ...
                 'accuracy', 10e-10, ...
                 'add_channels', 0, ...
                 'fs_orig', [], ...
                 'vars', {});

vers = version;
if (str2double(vers(1)) == 7)
  opt_save= {'-V6'};
else
  opt_save= {};
end

%% Check format of vars:
isch= apply_cellwise(opt.vars, 'ischar');
isch= [isch{:}];
if mod(length(opt.vars),2)~=0 || any(~isch(1:2:end)),
  error('wrong format for opt.vars');
end

%% Check for absolute paths:
%%  For Unix systems, absolute paths start with '\'.
%%  For Windoze, identify absolute paths by the ':' (e.g., H:\some\path).
if (isunix && (file(1)==filesep)) || (ispc && (file(2)==':')),
  [opt, isdefault]= opt_overrideIfDefault(opt, isdefault, 'path', '');
end

nChans= length(dat.clab);
fullname= [opt.path file];
if opt.add_channels && ~exist([fullname '.mat'], 'file'),
  warning('File does not exist: ignoring option ''add_channels''.');
  opt.add_channels= 0;
end

if opt.add_channels,
  if ~opt.channelwise,
    warning('add_channels requested: forcing channelwise mode');
    opt.channelwise= 1;
  end
  nfo_old= eegfile_loadMatlab(fullname, 'vars','nfo');
  if ~isdefault.format && opt.format~=nfo_old.format,
    warning(sprintf('format mismatch: using %s', nfo_old.format));
  end
  opt.format= nfo_old.format;
end

if ismember(upper(opt.format), {'INT16','AUTO'}),
  %% Look for resolutions producing lossless conversion to INT16.
  if strcmpi(opt.resolution, 'auto'),
    opt.resolution= NaN*ones(1, nChans);
    for cc= 1:nChans,
      rr= 0;
      res_found= 0;
      while ~res_found && rr<length(opt.resolution_list),
        rr= rr+1;
        X= dat.x(:,cc,:) / opt.resolution_list(rr);
        X= X(:);
        if all( abs(X-round(X)) < opt.accuracy ) && ...
              all(X>=-32768-opt.accuracy) && all(X<=32767+opt.accuracy),
          opt.resolution(cc)= opt.resolution_list(rr);
          res_found= 1;
        end
      end
      clear X;
    end
  end

  %% Expand global resolution.
  if length(opt.resolution)==1,
    opt.resolution= opt.resolution*ones(1,nChans);
  end

  %% Check format of resolution.
  if ~all(isnumeric(opt.resolution)) || length(opt.resolution)~=nChans,
    error('property resolution has invalid format');
  end

  %% If for all channels lossless conversions were found, auto-select 
  %% format INT16.
  if strcmpi(opt.format, 'auto'),
    if all(~isnan(opt.resolution)),
      opt.format= 'INT16';
    else
      opt.format= 'DOUBLE';
    end
  end
end

%% Check format of property format.
if ~ismember(upper(opt.format), {'INT16','FLOAT','DOUBLE'}),
  error('unknown format');
end

%% Select resolution for lossy conversion to INT16.
if strcmpi(opt.format, 'INT16'),
  iChoose= find(isnan(opt.resolution));
  for cc= 1:length(iChoose),
    ci= iChoose(cc);
    dat_ch= dat.x(:,ci,:);
    opt.resolution(ci)= 1.000001*max(abs(dat_ch(:)))'/32767;
  end
end

%% Gather some summary information into structure 'nfo'.
nfo= copy_struct(dat, 'fs', 'clab');
nfo.T= size(dat.x,1);
nfo.nEpochs= size(dat.x,3);
nfo.length= size(dat.x,1)*size(dat.x,3) / dat.fs;
nfo.format= opt.format;
nfo.resolution= opt.resolution;
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

%% if adding channels is requested, merge the nfo structures
if opt.add_channels,
  nfo.clab= cat(2, nfo_old.clab, nfo.clab);
  nfo.resolution= cat(2, nfo_old.resolution, nfo.resolution);
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

if opt.add_channels,
  %% update the nfo structure
  save(fullname, '-APPEND', 'nfo', opt_save{:});
  chan_offset= length(nfo_old.clab);
else
  save(fullname, 'mrk', 'mnt', 'nfo', opt_save{:});
  chan_offset= 0;
end

dat.file= fullname;
if opt.channelwise,
  rhs= 'dat.x(:,cc,:)';
  switch(upper(opt.format)), 
   case 'INT16',
    evalstr= ['int16(round(' rhs '/opt.resolution(cc)));'];
    dat.resolution= opt.resolution;
   case 'FLOAT',
    evalstr= ['float(' rhs ');'];
   case 'DOUBLE',
    evalstr= [rhs ';'];
  end
  for cc= 1:nChans,
    varname= ['ch' int2str(chan_offset+cc)];
    eval([varname '= ' evalstr]);
    save(fullname, '-APPEND', varname, opt_save{:});
    clear(varname);
  end
  dat= rmfield(dat, 'x');
  %% the following field updates are needed for add_channels=1
  dat.clab= nfo.clab;
  if ~ismember(upper(opt.format), {'FLOAT','DOUBLE'}),
    dat.resolution= nfo.resolution;
  end
  save(fullname, '-APPEND', 'dat', opt_save{:});
else
  switch(upper(opt.format)), 
   case 'INT16',
    for cc= 1:nChans,
      dat.x(:,cc,:)= round( dat.x(:,cc,:) / opt.resolution(cc) );
    end
    dat.x= int16(dat.x);
    dat.resolution= opt.resolution;
   case 'FLOAT',
    dat.x= float(dat.x);
   case 'DOUBLE',
    %% nothing to do
  end
  save(fullname, '-APPEND', 'dat', opt_save{:});
end

%% Save additional variables, as requested.
if ~isempty(opt.vars),
  for vv= 1:length(opt.vars)/2,
    eval([opt.vars{2*vv-1} '= opt.vars{2*vv};']);
  end
  save(fullname, '-APPEND', opt.vars{1:2:end}, opt_save{:});
end
