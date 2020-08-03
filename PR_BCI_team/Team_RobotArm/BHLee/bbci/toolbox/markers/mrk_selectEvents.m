function [mrk, ev]= mrk_selectEvents(mrk, varargin)

% MRK_SELECTEVENTS - selects events from the marker struct, provided either
% a vector of indices to choose, or criteria according to which
% events are chosen.
%
%Usage:
% mrk= mrk_selectEvents(mrk, ev, <OPTS>) - pick events indexed by ev
% mrk= mrk_selectEvents(mrk, <OPTS>)     - constrained selection of events
%                                          according to criteria
% IN:
% mrk:            usual mrk structure
% ev:             array with indices of events to select, also working for 1 single event.
%
%Normal picking: mrk_selectEvents(mrk, ev, <OPTS>)
% OPT - struct or property/value list of optional properties:
% 'random':         check whether length(ev) = 1 and pick randomly ev events from the marker structure
%                   (if it contains more than maxEvents events). (default
%                   0)
%
%Picking according to criteria: mrk_selectEvents(mrk, <OPTS>)
% OPT - struct or property/value list of optional properties:
% 'criterion'      - a cell array of criteria w.r.t. channels. Each
%                    criterion is a string and should correspond to a
%                    logical expression than can be evaluated. You can use
%                    channel names as shortcuts. For instance, 
%                    {'max(Fz)-min(Fz) > 100'} would select markers
%                    wherein the min-max difference for channel Fz exceeds
%                    100 in the corresponding interval. Default []. 
%                    If you provide criteria, you
%                    should also provide the cnt struct (see 'cnt').
% 'cnt'            - cnt struct that should be provided when criteria are
%                    set.
% 'criterionIval'  - markers falling within the time window (in ms) around the
%                    specified markers will be selected. If there are N
%                    invalid markers, invalidTime should be a 2 x N matrix.
%                    Default [-500 500] for each specified marker.
% 'class'          - cell array containing class names of markers or vector
%                    of the respective indices. All other markers falling
%                    within a specified interval around the markers are
%                    selected.
% 'classIval'      - markers falling within the time window (in ms) around the
%                    specified markers will be selected. If there are N
%                    invalid markers, invalidTime should be a 2 x N matrix.
%                    Default [-500 500] for each specified marker.
% 'keepSelected'   - if 1, all markers are kept but selected are marked 
%                    using an indexed field 'selectedMarkers' (default 0).
%
% General options:
% 'invert'        -  if 1, selection is inverted: all chosen markers are
%                    rejected, the rest is kept (default 0).
% 'sort'          -  evokes a call to mrk_sortChronologically (default 0).
% 'remainclasses' -  does not delete empty classes (default 0).
% 
% The structure 'mrk' may contain a field 'indexedByEpochs' being a
% cell array of field names of mrk. in this case subarrays of those
% fields are selected. Here it is assumed that the last dimension
% is indexed by events (resp. epochs).
%
%Examples:
% mrk = mrk_selectEvents(mrk, 1:10); % selects the first 10 events
% mrk = mrk_selectEvents(mrk, 1:10,'invert',1); % selects ALL BUT the first 10 events
%
% % Selects all events within 1000 ms around the event with classname 'event1'
% mrk = mrk_selectEvents(mrk, 'class','class1','classIval',...
% [-1000 1000]); 
%
% % Selects all events where Cz OR Fz is > 100 within 1000 ms around the event
% mrk = mrk_selectEvents(mrk, 'criterion','Cz > 100 | Fz > 100','criterionIval',...
% [-1000 1000]); 
% 
% bb 02/03, ida.first.fhg.de
% Matthias Treder Aug 2010: tidied up, added criterion-based selection & docu

%% Pre processing input
ev = NaN; 
if nargin==1,
  ev= find(any(mrk.y,1));    % pick all events
elseif isnumeric(varargin{1})
  ev = varargin{1};
  varargin(1) = [];
elseif islogical(varargin{1})
  ev = find(varargin{1});
  varargin(1) = [];
end

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'criterion', [], ...
                 'criterionIval', NaN, ...
                 'cnt', [], ...
                 'class', [], ...
                 'classIval', NaN, ...
                 'invert', 0,...
                 'keepSelected', 0, ...
                 'sort', 0, ...
                 'remainclasses', 0, ...
                 'random', 0);

% If class is given as strings, convert into indices
if ischar(opt.class) || iscell(opt.class)
  opt.class = find(ismember(mrk.className,opt.class));
end
if ischar(opt.criterion)
  opt.criterion = {opt.criterion};
end

% Set default ivals
defT = [-500 500]; % default ival
if isdefault.classIval && ~isempty(opt.class)
  opt.classIval = repmat(defT,[numel(opt.class) 1]);
end
if isdefault.criterionIval && ~isempty(opt.criterion)
  opt.criterionIval = repmat(defT,[numel(opt.criterion) 1]);
end
% Convert ivals from ms to samples
opt.criterionIval = round(opt.criterionIval/1000*mrk.fs);
opt.classIval = round(opt.classIval/1000*mrk.fs);
               
%% Pick and search
               
% Random picking?
if length(ev)==1 && ~isnan(ev) && opt.random==1
  maxEvents= ev;
  nEvents= length(mrk.pos);
  ev= randperm(nEvents);
  if nEvents>maxEvents,
   ev= ev(1:maxEvents);
  end
end

% Criterion selection
if isnan(ev)
%   idx = 1:length(mrk.pos); % all indices to check
  selected = false(1,length(mrk.pos)); % logical array indexing chosen elements

  % Markers
  if ~isempty(opt.class)
    for ii=1:numel(opt.class)
      cidx = find(mrk.y(opt.class(ii),:) & ~selected); % find all unmarked events for this class
      % Check ival around each relevant marker
      for jj=1:numel(cidx)
        ppp = mrk.pos - mrk.pos(cidx(jj));
        selected(ppp>opt.classIval(ii,1) & ppp<opt.classIval(ii,2))=1;
      end
    end
  end
  
  % Criteria
  if ~isempty(opt.criterion) 
    if isempty(opt.cnt), error('cnt must be provided.'), end
    for ii=1:numel(opt.criterion)
      fcn = opt.criterion{ii};
      % Replace channel names in function with 
      for cc=1:numel(opt.cnt.clab)
        fcn = strrep(fcn,opt.cnt.clab{cc},['opt.cnt.x(%,' num2str(cc) ')']);
        % '%' is a placeholder for the indices
      end
      cidx = find(~selected); % find all events that are not (yet) selected
      for jj=1:numel(cidx)
        % Check ival around each relevant marker
        cival = [mrk.pos(cidx(jj))+opt.criterionIval(ii,1) ...
          mrk.pos(cidx(jj))+opt.criterionIval(ii,2)];
        % Prevent falling out of the borders
        cival = [max(cival(1),1) min(cival(2),sum(opt.cnt.T))];
        cfcn = strrep(fcn,'%', ...
          ['[' num2str(cival(1)) ':' num2str(cival(2)) ']']);   % insert correct indices
        eval(['res=' cfcn ';']);
        if islogical(res)
          if numel(res)>1, res = any(res); end
        else
          error('Criterion function %s should return a logical value or vector',opt.criterion{ii})
        end
        selected(cidx(jj)) = res;
      end
    end
  end
  ev = find(selected);
end

%% Adapt marker struct
if opt.invert==1
  ev = setdiff(1:numel(mrk.pos),ev); 
  if exist('selected','var')
    selected = ~selected;
  end
end


% Fix toe and y fields
if ~opt.keepSelected
  mrk.pos= mrk.pos(ev);
  if isfield(mrk, 'toe'),
    mrk.toe= mrk.toe(ev);
  end
  if isfield(mrk, 'y'),
    mrk.y= mrk.y(:,ev);
  end
else
  % just add indexed field
  mrk.selectedMarkers = selected;
  mrk = mrk_addIndexedField(mrk,'selectedMarkers');
end

% Fix indexed epochs
if isfield(mrk, 'indexedByEpochs'),
  for Fld= mrk.indexedByEpochs,
    fld= Fld{1};
    tmp= getfield(mrk, fld);
    sz= size(tmp);
    subidx= repmat({':'}, 1, length(sz));
    subidx{end}= ev;
    mrk= setfield(mrk, fld, tmp(subidx{:}));
  end
end

%% Post process options
if opt.remainclasses==0 && isfield(mrk, 'y')
  mrk= mrk_removeVoidClasses(mrk);
end
if opt.sort==1
  if opt.remainclasses==0
    mrk= mrk_sortChronologically(mrk);
  else
    mrk= mrk_sortChronologically(mrk,0,'remainclasses');
  end
end

