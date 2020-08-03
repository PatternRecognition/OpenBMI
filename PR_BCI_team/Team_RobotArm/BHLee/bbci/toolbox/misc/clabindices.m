function ind= clabindices(lab, chans, varargin)
%CLABINDICES - Indices of channel labels
%
%Synopsis:
% IDX= clabindices(CLAB, CHANCELL, OPT)
%
%Arguments:
% CLAB: cell array of channel labels (or struct with field clab)
% CHANCELL: cell array of strings, where
%              each string is a channel label pattern as described here:.
%      channel label (string)  or only one chanx argument and
%              chan1 is a cell array of channel labels;
%              integer arguments are just returned;
%              in strings '#' matches numbers and 'z', 
%              and '*' matches everything (#/* must be the last symbol)
%              if the first string is 'not', ind will contain indices of
%              all channels except for the given ones.
% OPT: struct or property/value list of optinal properties:
%  'invert': invert the selection of channels, i.e. return the indices of
%     all channels which are NOT matched by the specified patterns.
%  'ignore_tail':
%
%Returns:
% IDX: indices of channels in channel enumeration given by the first argument
%
%Examples:
% clabindices(scalpChannels, {'CP3-4', 'C#', 'P3,z,4'})
%     matches CP3, CP1, CPz, CP2, CP4, C5, C3, C1, Cz,
%             C2, C4, C6, P3, Pz, P4
%
% clabindices(epo, 'not','E*');
%     matches all channels that do not start with letter 'E'

% bb, GMD-FIRST, 04/00

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'invert', 0, ...
                  'ignore_tails', 1);

tags={'9','7','5','3','1','z','2','4','6','8','10'};
if isstruct(lab), lab= lab.clab; end
if opt.ignore_tails,
  %% delete appendices (separated by a blank), e.g. 'C3 lap' -> 'C3'
  lab= strhead(lab);
end

if length(chans)>=1 & isequal(chans{1}, 'not'),
  opt.invert= 1;
  chans= chans(2:end);
end
if length(chans)==1,
  if iscell(chans{1}),
    chans= chans{1};
    if length(chans)>=1 & isequal(chans{1}, 'not'),
      if opt.invert,
        error('double negation is not allowed');
      end
      opt.invert= 1;
      chans= chans(2:end);
    end
  elseif isstruct(chans{1}) & isfield(chans{1}, 'clab'),
    chans= chans{1}.clab;
  elseif isnumeric(chans{1}),
    ind= chans{1};
    return;
  end
end
if isempty(chans) | (length(chans)==1 & isempty(chans{1}))
  if opt.invert,
    ind= 1:length(lab);
  else
    ind= [];
  end
  return;
end

% Choose unique channels only
[u, iUnique]=unique(chans);
chans=chans(sort(iUnique));

unknownChans= {};
nChans= length(chans);
ind= [];
for ch= 1:nChans,
  chanLab= chans{ch};
  if ischar(chanLab),
    if opt.ignore_tails,
      chanLab= strhead(chanLab);  %% arguable
    end
    iDash= find(chanLab=='-');
    if chanLab(end)=='#',
      new= strmatch(chanLab(1:end-1), lab)';
      for ni= 1:length(new),
        tail= lab{new(ni)}(length(chanLab):end);
        if isempty(strmatch(tail, tags, 'exact')),
          new(ni)= NaN;
        end
      end
      ind= [ind new(isfinite(new))];
    elseif ismember('*',chanLab),
      ind= [ind strpatternmatch(chanLab, lab)];
    elseif length(iDash)==1 & iDash<length(chanLab) & ...
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
          warning(sprintf('multiple channels of [%s] found.', chanLab));
          cc=cc';
        end
        ind= [ind cc];
      end
    end
  else
    ind= [ind chanLab];
  end
end

if opt.invert,
  ind= setdiff(1:length(lab), ind);
end

%if ~isempty(unknownChans),
%  warning('bci:missing_channels', ...
%          ['missing channels: ' vec2str(unknownChans)]);
%end
