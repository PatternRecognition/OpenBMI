function eegfile_writeBV(dat, mrk, varargin)
% EEGFILE_WRITEBV - Write EEG file in BrainVision format
%
% Synopsis:
%   eegfile_writeBV(dat, <mrk, scale>)
%   eegfile_writeBV(dat, mrk, 'Property1', Value1, ...)
%
% Arguments:
%   DAT: structure of continuous or epoch EEG data
%   MRK: marker structure as obtained by eegfile_readBVheader,
%        or simple marker format of bbci toolbox
%   SCALE: scaling factor used in the generic data format to bring
%          data from the int16 range (-32768 - 32767) to uV.
%          That means before saving signals are divided by
%          this factor.
%          Individual scaling factors may be specified for each
%          channel in a vector, or a global scaling as scalar,
%          default is 0.1 (i.e. signal range is -3276.8 - 3276.7).
%          scale can also be a precision in string format, such as
%          'float32'.
%
% Properties:
%   'scale': see arguments
%   'write_mrk': whether to write a marker file or not, default 1.
%   'write_hdr': whether to write a header file or not, default 1.
%   'export_dir': directory to which files are written, as default 
%                 the global variable EEG_EXPORT_DIR is taken
%   'filename': name of file, default dat.title. If this is not an
%               absolute path, opt.export_dir is prepended
%
% Remark: 
%   use 
%   scale= max(abs(cnt.x))'/32768;
%   to achive best resolution (least information loss in int16 conversion).
%
% See also: eegfile_*
%

if isfield(dat, 'title'),
  file= dat.title;
else
  file= '';
end

if length(varargin)==1 & ~isstruct(varargin{1}),
  opt= struct('scale',varargin{1});
else
  opt= propertylist2struct(varargin{:});
end

global EEG_EXPORT_DIR
opt= set_defaults(opt, ...
                  'write_mrk', 1, ...
                  'write_hdr', 1, ...
                  'export_dir', EEG_EXPORT_DIR, ...
                  'filename', file);

file= opt.filename;
if (isunix & file(1)==filesep) | (~isunix & file(2)==':')
  fullName= file;
else
  fullName= [opt.export_dir '/' file];
end

[T, nChans, nEpochs]= size(dat.x);
if nEpochs>1,
  cnt= permute(dat.x, [2 1 3]);
  cnt= reshape(cnt, [nChans T*nEpochs]);
  if ~exist('mrk','var'),
    nEpochs= floor(size(dat.x,1)/T);
    mrk= [];
    mrk.pos= (1:nEpochs)*T;
    mrk.y= ones(1,nEpochs);
  end
else
  nEpochs= 0;
  cnt= dat.x';
end

if isfield(dat,'scale'), 
  if isfield(opt, 'scale') & ~isempty(opt.scale),
    warning('optional property scale overwrites DAT.scale');
  else
    opt.scale= dat.scale;
  end
  opt.precision= 'int16';
else

  if isfield(opt, 'scale'),
    if ischar(opt.scale),   % scale is actually a number format.
      opt.precision= opt.scale;
      opt.scale= 1;
    else
      opt.precision= 'int16';
    end
  else 
    opt.scale= 0.1; 
    opt.precision= 'int16';
  end

  if length(opt.scale)==1, opt.scale= opt.scale*ones(nChans,1); end
  cnt= diag(1./opt.scale)*cnt;
  if any(cnt(:)>32767 | cnt(:)<-32768),
    warning('data clipped: use other scaling');
  end
end

subdir= fileparts(fullName);
if ~exist(subdir, 'dir'),
  parentdir= fileparts(subdir);
  if ~exist(parentdir, 'dir'),
    error('parent folder of %s not existing', subdir);
  end
  mkdir(subdir);
end
fid= fopen([fullName '.eeg'], 'wb');
if fid==-1, error(sprintf('cannot write to %s.eeg', fullName)); end
fwrite(fid, cnt, opt.precision);
fclose(fid);

if opt.write_hdr,
  opt_hdr= copy_struct(dat, 'fs','clab');
  opt_hdr.scale= opt.scale;
  opt_hdr.DataPoints= size(cnt,2);
  if ismember(opt.precision,{'float32','float','double','int16'})
    opt_hdr.precision = opt.precision;
  else
    error(['Unknown precision, not implemented yet: ' opt.precision]);
  end
  eegfile_writeBVheader(fullName, opt_hdr);
end

if opt.write_mrk,
  eegfile_writeBVmarkers(fullName, mrk);
end
