function mrk = nirsfile_readMarker(file, varargin)
% NIRSFILE_READMARKER - Read marker in NIRS format from single marker file.
%
% Synopsis:
%   [MRK, FS]= nirsfile_readMarker(MRKNAME,<OPT>)
%
% Arguments:
%   MRKNAME: name of marker file (no extension),
%            relative to NIRS_RAW_DIR (if it exists) or otherwise EEG_RAW_DIR
%            unless beginning with '/'
%
% Properties:
%   'system': NIRS system used (default 'nirx')
%   'prefix': string prepended to each marker string (default 'S '; e.g.
%             makes a marker "1" become "S 1")
%   'path' : directory with raw NIRS data (default NIRS_RAW_DIR (if exists)
%           or EEG_RAW_DIR). If data is file contains an absolute path,
%           path is ignored.

% Returns:
%   MRK: struct array of markers with fields desc (marker descriptor=number
%   from 0 to 15) and pos (position in samples relative to recording
%   onset).
%
% Description:
%   Read all marker information from a BrainVision generic data format file.
%   The sampling interval is read from the corresponding head file.
%
% Note: Based on eegfile_readBVmarkers.
% See also: nirs_* nirsfile_*
%
% matthias.treder@tu-berlin.de 2011

if ~isempty(whos('global','NIRS_RAW_DIR'))
  global NIRS_RAW_DIR
  nirsdir = NIRS_RAW_DIR;
else
  global EEG_RAW_DIR
  nirsdir = EEG_RAW_DIR;
end

if isabsolutepath(file)
  nirsdir = [];
end

opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, ...
                 'path',nirsdir, ...
                 'system','nirx',...
                 'prefix','S ', ...
                 'verbose', 0);
               
%% Scan marker file
fullName= fullfile(opt.path,file);
mrk = struct();

if strcmp(opt.system,'nirx')
  % Structure of file: (col 1) Timestamp (col 2-9) 8 bits signifying the
  % marker (caution! lowest bit is left [not right], need to flip bits before converting to decimal)
  fid = fopen([fullName '.evt'],'r');
  s= textscan(fid,'%d %s','delimiter','\n');
  pos = s{1}';
  % Remove \t's, flip bits and convert to decimal
  desc = apply_cellwise(s{2},inline('bin2dec(fliplr(strrep(x,sprintf(''\t''),'''')))','x'))';
  desc = apply_cellwise(desc,inline('sprintf(''%s%d'',y,x) ','x','y'),opt.prefix);

  mrk.pos = pos;
  mrk.desc = desc;
  mrk.type = repmat({'Stimulus'},1,numel(mrk.pos)); % Stimulus type reingehackt
  mrk = mrk_addIndexedField(mrk,{'type'  'desc'});
end

mrk.pos = double(mrk.pos);


