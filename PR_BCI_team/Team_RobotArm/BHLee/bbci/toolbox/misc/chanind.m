function ind= chanind(lab, varargin)
%ind= chanind(clab, chan1, <chan2, ...>)
%ind= chanind(dat, chan1, <chan2, ...>)
%ind= chanind(clab, chancell)
%
% IN   clab  - cell array of channel labels (or struct with field clab)
%              ! only the string up to the first space is used !
%      chanx - channel label (string)  or only one chanx argument and
%              chan1 is a cell array of channel labels;
%              integer arguments are just returned; 
%              the following special tokens can be used:
%              '#' matches 'z' or the number 1-10
%              '*' matches multiple numbers and/or letters
%              Any combination and number of these tokens can be used at
%              any location in the string.
%              If the first string is 'not', ind will contain indices of
%              all channels except for the given ones.
%      chancell - alternative input format: cell array of strings, where
%              each string is a channel label pattern as described above.
%
% Optional arguments:
%  ignore    - specify characters that should additionally match wildcards
%              in addition to alphanumeric characters. You have to
%              parametrize it as a cell array in the form like {'ignore' '-/\'} 
%              where '-/\' would be the set of characters to match the wildcards.
%
%
% OUT  ind   - indices of channels in channel enumeration given
%              by the first argument
%
% Xamples
%   chanind(scalpChannels, 'P#')
%     matches P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, Pz
%
%   chanind(scalpChannels, 'CP3-4', 'C#', 'P3,z,4')
%     matches CP3, CP1, CPz, CP2, CP4, C5, C3, C1, Cz,
%             C2, C4, C6, P3, Pz, P4
%
%   chanind(epo, 'not','E*');
%     matches all channels that do not start with letter 'E'

% bb, GMD-FIRST, 04/00, matthias.treder 11

tags={'9','7','5','3','1','z','2','4','6','8','10'};
if isstruct(lab), lab= lab.clab; end
%% delete appendices (separated by a blank), e.g. 'C3 lap' -> 'C3'
lab= strtok(lab);

chans= varargin;
if length(chans)==1 && isempty(chans{1}), ind= []; return; end

%% Search for 'ignore' parameter
ignore = [];
for ii=1:numel(chans)
  if iscell(chans{ii}) && ~isempty(chans{ii}) && strcmp(chans{ii}{1},'ignore')
    ignore = chans{ii}{2};
    chans(ii)=[];
    if ismember('\',ignore)
      % '\' needs to be escaped
      ignore = strrep(ignore,'\','\\');
    end
  end
end

%% Parse channels
if length(chans)>=1 && isequal(chans{1}, 'not'),
  INVERT= 1;
  chans= chans(2:end);
else
  INVERT= 0;
end
if length(chans)==1,
  if iscell(chans{1}),
    chans= chans{1};
    if length(chans)>=1 && isequal(chans{1}, 'not'),
      if INVERT,
        error('double negation is not allowed');
      end
      INVERT= 1;
      chans= chans(2:end);
    end
  elseif isstruct(chans{1}) && isfield(chans{1}, 'clab'),
    chans= chans{1}.clab;
  elseif isnumeric(chans{1}),
    ind= chans{1};
    return;
  end
end
if isempty(chans) || (length(chans)==1 && isempty(chans{1}))
  if INVERT,
    ind= 1:length(lab);
  else
    ind= [];
  end
  return;
end


%% Choose unique channels only
[u, iUnique]=unique(chans);
chans=chans(sort(iUnique));

unknownChans= {};
nChans= length(chans);
ind= [];
for ch= 1:nChans,
  chanLab= chans{ch};
  if ischar(chanLab),
    chanLab= strtok(chanLab);  %% arguable
    iDash= find(chanLab=='-');
    if ismember('*',chanLab) || ismember('#',chanLab)
      % Prepare regular expression pattern
      % Channel should match the beginning, '*' corresponds to \w*, '#' to
      % .
      tagstr = [cell2mat(strcat(tags(1:end-1),'|')) tags{end}];
      if isempty(ignore)
%         pat = ['^' strrep(strrep(chanLab,'#','.'),'*','\w*') '$'];  
        pat = ['^' strrep(strrep(chanLab,'#',['( ' tagstr ')']),'*','\w*') '$'];  
      else
        % Incorporate to-be-ignored characters
        pat = ['^' strrep(strrep(chanLab,'#',['[ ' tagstr ']']),'*',['[' ignore '\w]*']) '$'];  
      end
      mat = regexp(lab,pat);
      mat = cellfun(@isempty, mat, 'UniformOutput',0);
      ind= [ind  find(~[mat{:}])];
    elseif length(iDash)==1 && iDash<length(chanLab) && ...
          ismember(chanLab(iDash+1),'z123456789'),
      base= chanLab(1:iDash-2);
      from= strmatch(chanLab(iDash-1), tags, 'exact');
      to= strmatch(chanLab(iDash+1:end), tags, 'exact');
      expanded=  cellstr([repmat(base,to-from+1,1) char(tags{from:to})]);
      ind= [ind chanind(lab, expanded)];
    elseif ismember(',', chanLab),
      id= min(find(ismember(chanLab,'z123456789')));
      base= chanLab(1:id-1);
      list= strread(chanLab(id:end),'%s','delimiter',',');
      ll=length(list);
      expanded= cellstr([repmat(base,ll,1) char(list)]); 
      ind= [ind chanind(lab, expanded)];
    else
      cc= strmatch(chanLab, lab, 'exact');
      if isempty(cc),
        unknownChans= cat(2, unknownChans, {chanLab});
      else
        if length(cc)>1
          warning('bbci:multiple_channels', ...
            'multiple channels of [%s] found.', chanLab);
          cc=cc';
        end
        ind= [ind cc];
      end
    end
  else
    ind= [ind chanLab];
  end
end

if INVERT,
  ind= setdiff(1:length(lab), ind);
end

%if ~isempty(unknownChans),
%  warning('bbci:missing_channels', ...
%          ['missing channels: ' vec2str(unknownChans)]);
%end
