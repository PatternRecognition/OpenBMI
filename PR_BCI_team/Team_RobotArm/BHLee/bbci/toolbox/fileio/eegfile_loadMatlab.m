function [varargout]= eegfile_loadMatlab(file, varargin)
% EEGFILE_LOADMATLAB - Load EEG data structure from Matlab file
%
% Synopsis:
%   [DAT, MRK, MNT]= eegfile_loadMatlab(FILE, VARS)
%   [DAT, MRK, MNT]= eegfile_loadMatlab(FILE, 'Property1', Value1, ...)
%
% Arguments:
%   FILE: name of data file
%   VARS: Variables (cell array of strings) which are to be loaded,
%         default {'dat','mrk','mnt'}. The names 'dat', 'cnt' and 'epo'
%         are treated equally.
%
% Returns:
%   DAT: structure of continuous or epoched signals
%   MRK: marker structure
%   MNT: electrode montage structure
%
% Properties:
%   'vars': see arguments
%   'clab': Channel labels (cell array of strings) for loading a subset of
%           all channels. Default 'ALL' means all available channels.
%           See function 'chanind' for valid formats. In case OPT.clab is
%           not 'ALL' the electrode montage 'mnt' is adapted automatically.
%   'ival': Request only a subsegment to be read [msec]. This is especially
%           useful to load only parts of very large files.
%           Use [start_ms inf] to specify only the beginning.
%           Default [] meaning the whole time interval.
%   'fs': Sampling rate (must be a positive integer divisor of the fs
%         the data is saved in). Default [] meaning original fs.
%   'path': In case FILE does not include an absolute path, OPT.path
%           is prepended to FILE. Default EEG_MAT_DIR (global variable).
%
% Remark:
%   Properties 'ival' and 'fs' are particularly useful when data is saved
%   channelwise. Then cutting out the interval and subsampling is done
%   channelwise while reading the data, so a lot of memory is saved
%   compared to loading the whole data set first and then cutting out
%   the segment resp. subsample.
%
% Example:
%   file= 'Gabriel_03_05_21/selfpaced2sGabriel';
%   [cnt,mrk,mnt]= eegfile_loadMatlab(file);
%   %% or just to load variables 'mrk' and 'mnt':
%   [mrk,mnt]= eegfile_loadMatlab(file, {'mrk','mnt'});
%   %% or to load only some central channels
%   [cnt, mnt]= eegfile_loadMatlab(file, 'clab','C5-6', 'vars',{'cnt','mnt'});
%
% See also: eegfile_*
%
% Author(s): Benjamin Blankertz, Feb 2005
%

%% Warning: if the opt.ival option is used for *epoched* data,
%%   the field epo.t is not set correctly.

global EEG_MAT_DIR

if length(varargin)==1 & iscell(varargin{1}),
  opt= propertylist2struct('vars', varargin{1});
else
  opt= propertylist2struct(varargin{:});
end
default_vars= {'dat','mrk','mnt','nfo'};
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'path', EEG_MAT_DIR, ...
                 'clab', 'ALL', ...
                 'vars', default_vars(1:min(4,nargout)), ...
                 'ival', [], ...
                 'fs', []);

if ~iscell(opt.vars),
  opt.vars= {opt.vars};
end

if nargout~=length(opt.vars),
  warning('number of output arguments does not match with requested vars');
end

if iscell(file),
  varargout= cell(1, length(opt.vars));
  [varargout{:}]= eegfile_concatMatlab(file, opt);
  return;
end

%% Check for absolute paths:
%%  For Unix systems, absolute paths start with '\'.
%%  For Windoze, identify absolute paths by the ':' (e.g., H:\some\path).
if (isunix && (file(1)==filesep)) || (ispc && (file(2)==':')),
  if ~isdefault.path,
    warning('opt.path is ignored, since file is given with absolute path');
  end
  opt.path= '';
end
fullname= [opt.path file];

if ismember('*', file),
  [filepath, filename]= fileparts(fullname);
  dd= dir(filepath);
  resr= regexp({dd.name}, filename);
  cc= apply_cellwise2(resr, inline('~isempty(x)','x'));
  if sum(cc)==0,
    error(sprintf('no match for pattern ''%s'' in folder ''%s''', ...
                  filename, filepath));
  end
  iMatch= find(cc);
  fullname= strcat(filepath, '/', {dd(iMatch).name});
  varargout= cell(1, length(opt.vars));
  [varargout{:}]= eegfile_concatMatlab(fullname, opt);
  return;  
end

iData= find(ismember(opt.vars, {'dat','cnt','epo'}));

%% Load variables directly, except for data structure
load_vars= opt.vars;
load_vars(iData)= [];

%% Load non-data variables
S= load(fullname, load_vars{:});

%% Check whether all requested variables have been loaded.
missing= setdiff(load_vars, fieldnames(S));
if ~isempty(missing),
  error(['Variables not found: ' sprintf('%s ',missing{:})]);
end

%% Adapt electrode montage, if only a subset of channels is requested
if isfield(S, 'mnt') && ~isequal(opt.clab, 'ALL'),
  S.mnt= mnt_adaptMontage(S.mnt, opt.clab);
end

if ~isempty(opt.fs),
  load(fullname, 'nfo');
  lag= nfo.fs/opt.fs;
  if lag~=round(lag) || lag<1,
    error('fs must be a positive integer divisor of the file''s fs');
  end
else
  lag= 1;
end

if ~isempty(iData),
  wstat= warning('off');
  load(fullname, 'dat');
  warning(wstat);
  if ~exist('dat','var'),  %% old data file
    load(fullname, 'cnt');
    if ~exist('cnt','var'),
      error('neither variable <dat> nor <cnt> found.');
    end
    dat= cnt;
  end
  dat.file= fullname;
  if isfield(dat, 'x'),
    %% Data structure containing all channels was saved.
    if isequal(opt.clab, 'ALL'),
      chind= 1:size(dat.x,2);
    else
      chind= chanind(dat, opt.clab);
      dat= proc_selectChannels(dat, opt.clab);
    end
    dat.x= double(dat.x);
    if isfield(dat, 'resolution'),
      for ci= 1:length(chind),
        dat.x(:,ci,:)= dat.x(:,ci,:) * dat.resolution(chind(ci));
      end
    end
    if ~isempty(opt.ival),
      dat= proc_selectIval(dat, opt.ival);      
      dat.ival= opt.ival;
    end
    if ~isempty(opt.fs),
      dat= proc_subsampleByLag(dat, lag);
    end
  else
    %% Data has been saved channelwise.
    load(fullname, 'nfo');
    orig_clab= dat.clab;
    if isequal(opt.clab, 'ALL'),
      chind= 1:length(orig_clab);
    else
      chind= chanind(orig_clab, opt.clab);
    end
    dat.clab= orig_clab(chind);
    if ~isempty(opt.fs),
      lag= nfo.fs/opt.fs;
      if lag~=round(lag) || lag<1,
        error('fs must be a positive integer divisor of the file''s fs');
      end
      dat.fs= dat.fs/lag;
      if isfield(dat, 'T'),
        dat.T= dat.T./lag;
      end
    else
      lag= 1;
    end
    if ~isempty(opt.ival),
      dat.ival= opt.ival;
      iv= getIvalIndices(opt.ival, nfo);
      iOut= find(iv<1 | iv>nfo.T);
      if ~isempty(iOut),
        warning('requested interval too large: truncating');
        iv(iOut)= [];
      end
      if lag~=1,
        iv= iv(ceil(lag/2):lag:end);
      end
      T= length(iv);
      ivalstr= '(iv)';
    elseif lag>1,
      T= floor(nfo.T/lag);
      ival_start= ceil(lag/2);
      ivalstr= sprintf('(%d:%d:%d)', ival_start, lag, ival_start+lag*(T-1));
    else
      T= nfo.T;
      ivalstr= '';
    end
    dat.x= zeros(T, length(chind), nfo.nEpochs);
    for ci= 1:length(chind),
      varname= ['ch' int2str(chind(ci))];
      load(fullname, varname);
      dat.x(:,ci,:)= double(eval([varname ivalstr]));
      if isfield(dat, 'resolution'),
        dat.x(:,ci,:)= dat.x(:,ci,:) * dat.resolution(chind(ci));
      end
      clear(varname);
    end
    if isfield(dat, 'resolution'),
      dat= rmfield(dat, 'resolution');
    end
  end
end

%% cut back mrk structure to the requested interval
if isfield(S,'mrk'),
  if ~isempty(opt.fs),
    S.mrk.pos= ceil(S.mrk.pos/lag);
    S.mrk.fs= S.mrk.fs/lag;
    if isfield(S.mrk, 'T'),
      S.mrk.T= S.mrk.T./lag;
    end
  end
  if ~isempty(opt.ival),
    mrkpos_ms= S.mrk.pos/S.mrk.fs*1000;
    inival= find(mrkpos_ms>=opt.ival(1) & mrkpos_ms<=opt.ival(2));
    S.mrk= mrk_chooseEvents(S.mrk, inival);
    S.mrk.pos= S.mrk.pos - opt.ival(1)/1000*S.mrk.fs;
    S.mrk.ival= opt.ival;
  end
end

if ~isempty(opt.fs), % do resampling in nfo
  S.nfo.fs = opt.fs;
  S.nfo.T = nfo.T./lag;
end
for vv= 1:nargout,
  if ismember(vv, iData),
    varargout(vv)= {dat};
  else
    varargout(vv)= {getfield(S, opt.vars{vv})};
  end
end
