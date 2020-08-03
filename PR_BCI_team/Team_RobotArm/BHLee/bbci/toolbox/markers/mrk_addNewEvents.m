function mrk = mrk_addNewEvents(mrk, class, pos, varargin)

% MRK_ADDNEWEVENTS - Adds new event markers to an existing marker
% structure. Indexed-by-epochs fields are padded with the value of their
% nearest neighbour or a pre-specified value. Events are added to the end.
%
%Usage: mrk= mrk_addNewEvents(mrk, class, pos)
%
%
% IN:  mrk       - marker structure
%      class     - class name(s) or class index(indices). Multiple class 
%                  names should be provided as a cell array of strings.
%                  Class names may be new.
%      pos       - positions of the events as vector. In case of multiple 
%                  classes, a cell array of vectors should be provided.
%
% OPT - struct or property/value list of optional properties:
%  toe            - if new classes are provided, toe values can be given.
%                   By default, the smallest free markers are taken.
%  chronological  - if 1, events are added so that they are in
%                   chronological order. This is achieved via a call to 
%                   mrk_sortChronologically (default 1).
%                   
%  indexed        - if 'nearest', indexed fields are padded with the
%                   nearest neighbour. Alternatively, a value can be
%                   provided that is used for padding.
%
% OUT: mrk       - updated marker structure

% Matthias Treder Aug 2010

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'toe',[], ...
                 'chronological',1, ...
                 'indexed', 'nearest');
               
if ~iscell(class), class = {class}; end
if ~iscell(pos), pos = {pos}; end
if isdefault.toe 
  opt.toe = zeros(1,numel(class));
end

% Add events
for ii=1:numel(class)
  if ischar(class{ii})
    cIdx = find(ismember(mrk.className,class{ii}), 1); % class index
  else
    cIdx = class{ii};
  end
  if isempty(cIdx)
    % Add new class
    if ~iscell(mrk.className), mrk.className = {mrk.className};  end
    mrk.className = {mrk.className{:} class{ii}};
    fprintf('Added new class %s\n',class{ii});
    cIdx = numel(mrk.className);
    if isdefault.toe
      cToe = min(setdiff(1:256, unique(mrk.toe)));
    else
      cToe = opt.toe(ii);
    end;
  else
    cToe = unique(mrk.toe(mrk.y(cIdx,:)>0));
  end
  cPos = pos{ii};
  np = numel(pos{ii});
  mrk.pos = [mrk.pos cPos];
  mrk.toe = [mrk.toe cToe*ones(1,np)];
  mrk.y(cIdx,end+1:end+np) = 1;
  % Padding indexed fields
  if isfield(mrk, 'indexedByEpochs'),
    for Fld= mrk.indexedByEpochs,
      fld= Fld{1};
      if strcmp(opt.indexed,'nearest')
        % Find nearest neighbours
        for jj=1:numel(cPos)
          [dummy,nidx] = min(mrk.pos-cPos(jj));
          mrk.(fld) = [mrk.(fld) mrk.(fld)(nidx)];
        end
      else
        mrk.(fld) = [mrk.(fld) opt.indexed*ones(1,np)];
      end
    end
  end
end

% Sort it
if opt.chronological
  mrk = mrk_sortChronologically(mrk);
end