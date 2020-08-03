function mnt = nirs_restrictMontage(mnt,varargin)
% NIRS_RESTRICTMONTAGE - restricts the NIRS montage by selecting (a)
%       specified NIRS channels and/or (b) sources or detectors and 
%       keeping only the corresponding NIRS channels AND/OR (c) keeping only 
%       informative NIRS channels (ie channels corresponding to a relatively 
%       small source-detector distance).
%
% Synopsis:
%   MNT = nirs_restrictMontage(MNT, CHANS, <OPT>)
%   MNT = nirs_restrictMontage(MNT, <OPT>)
%
% Properties:
%    mnt             -  NIRS montage
%    chans           -  restrict montage to these NIRS channels (can
%                       contain any wildcards * and #; see chanind). 
%                       Must be a cell array of strings.
%
% OPT - struct or property/value list of optional properties:
%   'source':  a cell array containing the labels or physical numbers of
%              sources that are to be selected. All other sources and NIRS
%              channels containing these sources are removed. Default {}
%              (ie all sources are considered).
%              The cell array can also contain the keyword 'not' as first
%              element, in which the selection is inverted (ie {'not'
%              'Fz'} would remove Fz and preserve all *other* sources).
%   'detector':  The same as 'source' for the detectors.
%   'dist'  :  gives the maximum distance [in cm] between source/detector pairs. 
%              If set, source/detector pairs with a larger distance are
%              removed.  Assuming a head radius of default 10 cm (set 'headRadius').
%              The default value of dist is 3.5 - if you do not want
%              channels to be reduced at all according to distance, set
%              dist to [].
%   'removeOptodes' : if 1, the non-selected optodes are not only removed
%              from the NIRS channels (mnt.clab field) but also from the
%              corresponding source and detector fields (mnt.source and
%              mnt.detector). (default 1)
%
% OUT:  mnt             -  updated montage
%
% Note: Use proc_selectChannels to reduce the NIRS data (cnt,dat,epo) 
% according to the new montage.
%
% See also: nirs_getMontage, mnt_restrictMontage
%
% matthias.treder@tu-berlin 2011

%% Check which variant of the function is used
if nargin>1 && mod(nargin,2)==0 %% first varargin is CHANS
  if numel(varargin)>1
    opt = propertylist2struct(varargin{2:end});
    opt.chans = varargin{1};
  elseif isstruct(varargin{1})
    opt = varargin{1};
  else
    opt = struct();
    opt.chans = varargin{1};
  end
else
  opt = propertylist2struct(varargin{:});
end

opt = set_defaults(opt, ...
                  'chans',{}, ...
                  'source',{},'detector',{},...
                  'removeOptodes', 1, ...
                  'headRadius',10, ...
                  'dist', 3.5);

if ischar(opt.source)
  opt.source = {opt.source};
end
if ischar(opt.detector)
  opt.detector = {opt.detector};
end
if ischar(opt.chans)
  opt.chans = {opt.chans};
end
                
%% Select sources and/or detectors
if ~isempty(opt.source)
  selSou = mnt.source.clab(chanind(mnt.source,opt.source));
  if opt.removeOptodes
    mnt.source = mnt_restrictMontage(mnt.source,selSou);
  end
end
if ~isempty(opt.detector)
  selDet = mnt.detector.clab(chanind(mnt.detector,opt.detector));
  if opt.removeOptodes
    mnt.detector = mnt_restrictMontage(mnt.detector,selDet);
  end
end

%% Find connector for source-detector labels (non-alphanumeric character)
str = strhead(mnt.clab);
[a,a,a,connector] = regexp(str,'[^\w]');
connector = cell2mat(unique(cell_flaten(connector)));
if isempty(connector)
  connector = '';
elseif numel(connector)>1
  error('Multiple connectors in NIRS clab: [%s]',[connector{:}])
end

%% Select specified NIRS channels
if ~isempty(opt.chans)
  mnt = mnt_restrictMontage(mnt,opt.chans,{'ignore' connector}); 
end

%% Restrict NIRS channels by removing the deleted sources/detectors
if ~isempty(opt.source) || ~isempty(opt.detector)
  if isempty(opt.source), selSou = '*'; end
  if isempty(opt.detector), selDet = '*'; end
  if ~iscell(selSou), selSou = {selSou}; end
  % Build selection string
  sel = strcat(selSou,connector);
  sel = cell_flaten(apply_cellwise(sel,'strcat',selDet));
  mnt = mnt_restrictMontage(mnt,sel);
end


%% Reduce NIRS channels according to source-detector distance
if ~isempty(opt.dist)
  dist = mnt.angulardist * opt.headRadius;   % distances in cm
  sel = find(dist<opt.dist);  
  mnt = mnt_restrictMontage(mnt,sel);
end
