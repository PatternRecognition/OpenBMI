function [Mrk, fs]= eegfile_readBVmarkers(mrkName, varargin)
% EEGFILE_READBVMARKERS - Read Markers in BrainVision Format
%
% Synopsis:
%   [MRK, FS]= eegfile_readBVmarkers(MRKNAME, <OPT>)
%   [MRK, FS]= eegfile_readBVmarkers(MRKNAME, <OUTPUTSTRUCTARRAY>)
%
% Arguments:
%   MRKNAME - name of marker file (no extension),
%            relative to EEG_RAW_DIR unless beginning with '/'
%   OPT - Struct or property/value list of optional properties:
%     .outputStructArray: specifies the output format, default 0.
%                      If false the output is a struct of arrays, not
%                      a struct array.
%     .marker_format:  The format of the field MRK.desc:
%                          'string' : e.g. 'R  1', 'S123'
%                          'numeric': e.g.    -1 ,   123
%                          (default: 'string')
% 
%   OUTPUTSTRUCTARRAY - As OPT.outputStructArray
%
% Returns:
%   MRK: struct array of markers with fields
%        type, desc, pos, length, chan, time
%        which are defined in the BrainVision generic data format,
%        see the comment lines in any .vmrk marker file.
%   FS: samling rate, as read from the corresponding header file
%
% Description:
%   Read all marker information from a BrainVision generic data format file.
%   The sampling interval is read from the corresponding head file.
%
% See also: eegfile_*

% blanker@cs.tu-berlin.de


global EEG_RAW_DIR

if length(varargin)==1 && isnumeric(varargin{1}),
  opt= struct('outputStructArray', varargin{1});
else
  opt= propertylist2struct(varargin{:});
end
opt= set_defaults(opt, ...
                  'outputStructArray', 0, ...
                  'marker_format', 'string');

if (isunix & mrkName(1)==filesep) | (~isunix & mrkName(2)==':')
  fullName= mrkName;
else
  fullName= [EEG_RAW_DIR mrkName];
end


s= textread([fullName '.vmrk'],'%s','delimiter','\n');
skip= strmatch('[Marker Infos]', s, 'exact')+1;
if skip<=length(s)
  while s{skip}(1)==';',
    skip= skip+1;
  end
end 
opt_read= {'delimiter',',', 'headerlines',skip-1};

[mrkno,Mrk.type,Mrk.desc,Mrk.pos,Mrk.length,Mrk.chan,Mrk.clock]= ...
    textread([fullName '.vmrk'], 'Mk%u=%s%s%u%u%u%s', opt_read{:});
if strcmp(opt.marker_format, 'numeric'),
  [toe,idx]= marker_mapping_SposRneg(Mrk.desc);
  Mrk.desc= zeros(size(Mrk.desc));
  Mrk.desc(idx)= toe;
end

keyword= 'SamplingInterval';
s= textread([fullName '.vhdr'],'%s','delimiter','\n');
ii= strmatch([keyword '='], s);
fs= 1000000/sscanf(s{ii}, [keyword '=%f']);
if(fs ~= round(fs))
  warning('eegfile_readBVmarker: fs was not a whole number: %f',fs); 
  fs= round(fs);
end

if opt.outputStructArray,
  mrk= Mrk;
  Mrk= struct('type',mrk.type, 'desc',mrk.desc, 'pos',num2cell(mrk.pos), ...
              'length',num2cell(mrk.length), 'chan',num2cell(mrk.chan), ...
              'clock',mrk.clock);
else
  Mrk.pos= emptycheck(Mrk.pos');
  Mrk.type= emptycheck(Mrk.type');
  Mrk.desc= emptycheck(Mrk.desc');
  Mrk.length= emptycheck(Mrk.length');
  Mrk.chan= emptycheck(Mrk.chan');
  Mrk.clock= emptycheck(Mrk.clock');
  Mrk.indexedByEpochs= {'type','desc','length','clock','chan'};
  Mrk.fs= fs;
end
if nargout<=1,
  clear fs;
end



%% BB: ? what is this for ?
function x= emptycheck(x)

if isempty(x),
  if iscell(x),
    x= {};
  elseif ischar(x),
    x= '';
  else
    x= [];
  end
end
