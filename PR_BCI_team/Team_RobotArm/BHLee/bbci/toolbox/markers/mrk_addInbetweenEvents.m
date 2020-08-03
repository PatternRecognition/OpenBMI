function mrk_out= mrk_addInbetweenEvents(mrk, time_dist, varargin)

% MRK_ADDINBETWEENEVENTS - Adds markers for 'rest' events inbeween given 
% markers when time distances are big enough.
%
%Usage: mrk= mrk_addInbetweenEvents(mrk, time_dist, <OPT>)
%
%
% IN:  mrk       - marker structure
%      time_dist - vector of length 3: [td_before, td_between, td_after]
%                  defining the time distance to preceding events (td_before),
%                  between rest events (td_between) and to following events
%                  (td_after), all in msec.
%                  if time_dist is scalar all three time distance parameters
%                  are set to this value.
%
% OPT - struct or property/value list of optional properties:
%  extract_ival   - each row defines an interval in which markers are
%                   extracted, default all.
%  className      - name of the class of new markers, default 'rest'
%  between_what   - vector of marker types (as in mrk.toe) or cell array of
%                   class names which are regarded when placing the
%                   in-between markers (default all).
%
% OUT: mrk       - updated marker structure

% bb 08/2003, ida.first.fhg

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'className','rest', ...
                 'extract_ival',[],...
                 'between_what',unique(mrk.toe), ...
                 'addAcrossRuns',1);
               

if ~isdefault.between_what && iscell(opt.between_what)
  % Translate class names into toe values
  idx = find(ismember(mrk.className,opt.between_what));
  opt.between_what = unique(mrk.toe(any(mrk.y(idx,:))));
end

if length(time_dist)==1, time_dist= time_dist([1 1 1]); end
if length(time_dist)~=3, error('time_dist vector must have length 1 or 3'); end
% if ~exist('extract_ival', 'var') extract_ival=[]; end
% if ~exist('className','var') | isempty(className), className='rest'; end
% if ~exist('between_what','var'), between_what= unique(mrk.toe); end


rest_marker= min(setdiff(1:256, unique(mrk.toe)));
min_dist= max(time_dist([1 3]));
relevant_markers= find(ismember(mrk.toe, opt.between_what));

if ~isempty(opt.extract_ival)
  nIvals= size(opt.extract_ival,1);
  extract_ok= zeros(nIvals, length(relevant_markers));
  for ii= 1:nIvals,
    extract_ok(ii,:)= (mrk.pos(relevant_markers)>=opt.extract_ival(ii,1) & ...
                       mrk.pos(relevant_markers)<=opt.extract_ival(ii,2));
  end
  really_relevant= find(any(extract_ok,1));
  relevant_markers= relevant_markers(really_relevant);
end
dd= diff(mrk.pos(relevant_markers))*1000/mrk.fs;

td= round(time_dist/1000*mrk.fs);
iSpace= find(dd>min_dist);
mrk_rest.pos= [];
for ii= 1:length(iSpace),
  is= iSpace(ii);
  new_pos= mrk.pos(relevant_markers(is)) + td(1) : ...
           td(2) : ...
           mrk.pos(relevant_markers(is+1)) - td(3);
  mrk_rest.pos= cat(2, mrk_rest.pos, new_pos);
end

%% take only markers whose position falls into one of the
%% given extraction intervals
if ~isempty(opt.extract_ival),
  nIvals= size(opt.extract_ival,1);
  extract_ok= zeros(nIvals, length(mrk_rest.pos));
  for ii= 1:nIvals,
    extract_ok(ii,:)= (mrk_rest.pos>=opt.extract_ival(ii,1) & ...
                       mrk_rest.pos<=opt.extract_ival(ii,2));
  end
  iExtract= find(any(extract_ok,1));
  mrk_rest.pos= mrk_rest.pos(iExtract);
end

mrk_rest.fs= mrk.fs;
mrk_rest.toe= rest_marker*ones(size(mrk_rest.pos));
mrk_rest.y= ones(size(mrk_rest.pos));
mrk_rest.className= {opt.className};

mrk_out= mrk_mergeMarkers(mrk, mrk_rest);
