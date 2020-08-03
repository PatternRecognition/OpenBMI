function eegfile_splitBV(file, varargin)
% EEGFILE_SPLITBV - Split BrainVision EEG files
%
% Synopsis:
%   eegfile_splitBV(FILE, OPT);
%   eegfile_splitBV(FILE, MAXSIZE);
%
% Arguments:
%   FILE: name of the BV file (no extension)
%   MAXSIZE: max size of splitted files [MB], default 500.
%
% Properties:
%   'maxsize': See MAXSIZE
%   'appendix': For the use in sprintf(..., file_no) or product a string,
%               which is appended to the splitted files for numbering, 
%               (default '_pt%02d')
%   other options are passed to eegfile_loadBV
%
% See also: eegfile_*
%
% Author(s): blanker@cs.tu-berlin.de 05-2007
%

global EEG_RAW_DIR

if length(varargin)==1 & ~isstruct(varargin{1}),
  opt= struct('maxsize', varargin{1});
else
  opt= propertylist2struct(varargin{:});
end
opt= set_defaults(opt, ...
                  'maxsize',500, ...
                  'appendix', '_pt%02d', ...
                  'clab', '*', ...
                  'fs', 'raw', ...
                  'verbose', 1);

hdr= eegfile_readBVheader(file);

nChans= length(clabindices(hdr, opt.clab));
if isequal(opt.fs, 'raw'),
  opt.fs= hdr.fs;
end
mb_per_sec= opt.fs*nChans*2 / (1024*1024);
dur_sec= opt.maxsize/mb_per_sec;
dur_sa= floor(dur_sec*opt.fs);
if opt.verbose,
  fprintf('Splitting file %s.\n', file);
  fprintf(['The file uses %.2f MB per sec, ' ...
           'each new file will be %d sec long\n'], ...
          mb_per_sec, round(dur_sec));
end

mrk= eegfile_readBVmarkers(file);
mrkpos= [mrk.pos];
fileno= 0;
ptr= 1;
while 1,
  ival_sa= ptr+[0 dur_sa];
  clear cnt
  cnt= eegfile_loadBV(file, opt, 'prec',1, 'ival_sa',ival_sa);
  if isempty(cnt),
    break;
  end
  fileno= fileno+1;
  idx= find(mrkpos>=ival_sa(1) & mrkpos<=ival_sa(2));
  save_name= sprintf(['%s' opt.appendix], file, fileno);
  cnt.title= save_name;
  if opt.verbose,
    fprintf('saving samples %.0f to %.0f to %s.\n', ival_sa, save_name);
  end
  eegfile_writeBV(cnt, mrk(idx), 'filename',[EEG_RAW_DIR save_name]);
  ptr= ptr + dur_sa + 1;
end
