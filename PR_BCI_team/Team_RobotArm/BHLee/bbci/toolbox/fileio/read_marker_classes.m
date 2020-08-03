function mrk= read_marker_classes(mrkName, fs, classDef, ival, markerTypes)
%mrk= read_marker_classes(mrkName, <fs=100, classDef, ival, markerTypes>)
%
% IN   mrkName    - name of marker file (no extension),
%                   relative to EEG_RAW_DIR unless beginning with '/'
%      fs         - calculate marker positions for sampling rate fs,
%                   default: 100
%      classDef   - cell array { 1 or 2 x nClasses }, in the first row
%                   each entry (string or cell array) defines a class
%                   by the description of the markers that belong to that 
%                   class. E.g., {{'R65','R70'},{'R74','R192'},'S*'} defines
%                   three classes, the first including markers R65 and R70,
%                   the second including markers R74 and R192 and the third
%                   including all markers starting with 'S'.
%                   In an optional 2nd row classNames can be defined,
%                   default {'S*','R*'; 'stimulus','response'}
%      ival       - interval [start_msec end_msec] for which markers are read,
%                   default [0 inf], i.e., read all markers
%      markerTypes- string or cell array, only markers of those types are 
%                   considered, * is allowed as wildcard as first or last 
%                   symbol, default '*'
%
% OUT  mrk        struct for event markers
%         .toe    - type of event
%         .pos    - position in data points (for lagged data)
%         .fs     - sampling interval
%
% C readMarkerComment
%
% GLOBZ  EEG_RAW_DIR

%% 02/2004 blanker@first.fhg.de (ida)

global EEG_RAW_DIR

if ~exist('fs', 'var'), fs=100; end
if ~exist('classDef', 'var'), classDef={'S*','R*'; 'stimulus','response'}; end
if ~exist('ival', 'var'), ival=[0 inf]; end
if ~exist('markerTypes', 'var'), markerTypes='*'; end

if mrkName(1)==filesep,
  fullName= mrkName;
else
  fullName= [EEG_RAW_DIR mrkName];
end

keyword= 'SamplingInterval';
%try
s= textread([fullName '.vhdr'],'%s','delimiter','\n');
%catch, error(sprintf('%s.vhdr not found', fullName)); end
ii= strmatch([keyword '='], s);
mrk_fs= 1000000/sscanf(s{ii}, [keyword '=%d']);
iv= ival/1000*fs;

%try
s= textread([fullName '.vmrk'],'%s','delimiter','\n');
%catch, error(sprintf('%s.vmrk not found', fullName)); end

skip= strmatch('[Marker Infos]', s, 'exact')+1;
while s{skip}(1)==';',
  skip= skip+1;
end
opt= {'delimiter',',', 'headerlines',skip};

[mrkno,mrktype,desc,pos,pnts,chan,seg_time]= ...
    textread([fullName '.vmrk'], 'Mk%u=%s%s%u%u%u%s', opt{:});
valid= strpatternmatch(markerTypes, mrktype);
pos= ceil(pos(valid)*fs/mrk_fs) - iv(1);
desc= desc(valid);

nClasses= size(classDef, 2);
mrk.pos= [];
mrk.y= zeros(nClasses, 0);
for cc= 1:nClasses,
  ind= strpatternmatch(classDef{1,cc}, desc);
  mrk.pos= cat(2, mrk.pos, pos(ind)');
  lab= zeros(nClasses, length(ind));
  lab(cc,:)= 1;
  mrk.y= cat(2, mrk.y, lab);
end
mrk.fs= fs;

if size(classDef,1)==2,
  mrk.className= classDef(2,:);
end

valid= find(mrk.pos>0 & mrk.pos<=diff(iv));
mrk= mrk_selectEvents(mrk, valid);
