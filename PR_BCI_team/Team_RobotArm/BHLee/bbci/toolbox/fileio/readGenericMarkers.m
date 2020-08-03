function [Mrk, fs]= readGenericMarkers(mrkName, outputStructArray)
%[Mrk, fs]= readGenericMarkers(fileName, <outputStructArray>)
%
% Read all marker information from a BrainVision generic data format file.
% The sampling interval is read from the corresponding head file.
%
% IN:  mrkName  - name of marker file (no extension),
%                 relative to EEG_RAW_DIR unless beginning with '/'
%      outputStructArray - specifies the output format, default 1.
%                 If false the output is a struct of arrays, not
%                 a struct array.
%
% OUT: Mrk - struct array of markers with fields
%            type, desc, pos, length, chan, time
%            which are defined in the BrainVision generic data format,
%            see the comment lines in any .vmrk marker file.
%      fs  - samling rate, as read from the corresponding header file
%
%     
global EEG_RAW_DIR

if nargin<2, outputStructArray=1; end


if mrkName(1)==filesep,
  fullName= mrkName;
else
  fullName= [EEG_RAW_DIR mrkName];
end


s= textread([fullName '.vmrk'],'%s','delimiter','\n');
skip= strmatch('[Marker Infos]', s, 'exact')+1;
while s{skip}(1)==';',
  skip= skip+1;
end
opt= {'delimiter',',', 'headerlines',skip-1};

[mrkno,Mrk.type,Mrk.desc,Mrk.pos,Mrk.length,Mrk.chan,Mrk.time]= ...
    textread([fullName '.vmrk'], 'Mk%u=%s%s%u%u%u%s', opt{:});

if ~outputStructArray | nargout>1,
  keyword= 'SamplingInterval';
  s= textread([fullName '.vhdr'],'%s','delimiter','\n');
  ii= strmatch([keyword '='], s);
  Mrk.fs= 1000000/sscanf(s{ii}, [keyword '=%d']);
end

if outputStructArray,
  mrk= Mrk;
  Mrk= struct('type',mrk.type, 'desc',mrk.desc, 'pos',num2cell(mrk.pos), ...
              'length',num2cell(mrk.length), 'chan',num2cell(mrk.chan), ...
              'time',mrk.time);
  if nargout>1,
    fs= mrk.fs;
  end
end
