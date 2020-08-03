function [epo, complete]= proc_segmentation(cnt, mrk, ival, varargin)
%PROC_SEGMENTATION - Segment continuous signals into epochs
%
%Synopsis:
%  [EPO, IDX]= proc_segmentation(CNT, MRK, IVAL, <OPT>)
%  [EPO, IDX]= proc_segmentation(CNT, TIME, IVAL, <OPT>)
%
%Arguments:
%  CNT:  STRUCT       - Continuous 1D or 2D data (see file_readBV, file_loadMatlab)
%  MRK:  STRUCT       - Marker structure, with obligatory field 'time',
%                       which specifies the time points which define the t=0
%                       for each segment that is cut out from the continuous
%                       signals. 
%  TIME: DOUBLE       - Can be provided as alternative to MRK.time.
%  IVAL: DOUBLE [M 2] - Time interval relative to marker [start ms, end ms].
%                       If M>1 it must be equal to the number of classes, in
%                       which case it defines a unique interval for each class.
%                       Classes are define in MRK.y
%  OPT:  PROPLIST     - Struct or property/value list of optional properties:
%    'CLab': CELL or '*' (default)  - selection of channels {channel names}.
%                       '*' selects all channels.
%    'DiscardIncompleteSegments': BOOL
%
%Returns:
%  EPO -  structure of epoched signals
%    .fs    - sampling interval
%    .x     - signals (time x channels x epochs)
%    .clab  - channel labels
%    .t     - time axis
%    .y     - class labels (if it is a field of MRK)
%    .className - class names (if it is a field of MRK)
%    .cnt_info - fields other than 'x', 'clab', 'fs' are copied from cnt
%                into this subfield
%    .mrk_info - fields other than 'time', 'y', 'className', 'event' are copied
%                from mrk into this subfield
%
%  IDX - Indices of the markers that have been transformed to segments.
%     In case, some segments cannot be retrieved, since they would exceed
%     the borders of CNT this return argument can be of interest:
%     IDX is the vector of indices of markers that have been used for
%     segmentation (i.e., indices of the discarded epochs are left out).
%
%Description:
%  This function takes the continuous EEG data (as loaded by for instance 
%  file_readBV) and converts it into data, segmented around the markers
%  as given by the MRK struct.
% 
%Examples:
%  [cnt, mrk]= file_readBV(some_file);
%  mrk= mrk_defineClasses(mrk, {1, 2; 'target','nontarget'});
%  epo= proc_segmentation(cnt, mrk, [-200 800], 'CLab', {'Fz','Cz','Pz'});
%
%  % Different intervals for different classes:
%  mrk= mrk_defineClasses(mrk, {1, 1; 'pre-stim','post-stim'});
%  epo= proc_segmentation(cnt, mrk, [-1000 0; 0 1000]);
%
%See also:  file_readBV, file_loadMatlab, mrk_defineClasses.

% 02-2009 Benjamin Blankertz


props= {'CLab'                       '*'       'CHAR|CELL{CHAR}'
        'DiscardIncompleteSegments'  1,        'BOOL'};

if nargin==0,
  epo= props; return
end

misc_checkType(cnt, 'STRUCT(x clab fs)');
misc_checkType(cnt.x, 'DOUBLE[- -]|DOUBLE[- - -]', 'cnt.x');
misc_checkType(cnt.clab, 'CELL{CHAR}', 'cnt.clab');
misc_checkType(cnt.fs, 'DOUBLE[1]', 'cnt.fs');
misc_checkType(mrk, 'DOUBLE[-]|STRUCT(time)');
misc_checkType(ival, 'DOUBLE[- 2]');

opt= opt_proplistToStruct(varargin{:});
[opt, isdefault]= opt_setDefaults(opt, props);
opt_checkProplist(opt, props);
cnt = misc_history(cnt);

if ~isstruct(mrk), 
  mrk= struct('time', mrk);
end

%% redefine marker positions in case different intervals are
%% requested for different classes
if size(ival,1)>1,
  if ~isfield(mrk, 'y'),
    error('Different interval may only be specified, if MRK has a field ''y''.');
  end
  if size(ival,1)~=size(mrk.y,1),
    error('#rows of IVAL does not match #classes of MRK');
  end
  dd= diff(ival, 1, 2);
  if any(dd~=dd(1)),
    error('Intervals must all be of the same length for all classes');
  end
  %% set mrk.time such that all requested intervals are [-len 0]
  for cc= 1:size(ival,1),
    shift= ival(cc,2);
    idx= find(mrk.y(cc,:));
    mrk.time(idx)= mrk.time(idx) + shift;
  end
  ival= [-diff(ival(1,:)) 0];
end

si= 1000/cnt.fs;
TIMEEPS= si/100;
nMarkers= length(mrk.time);
len_sa= round(diff(ival)/si);
pos_zero= ceil((mrk.time-TIMEEPS)/si);
core_ival= [ceil(ival(1)/si) floor(ival(2)/si)];
addone= diff(core_ival)+1 < len_sa;
pos_end= pos_zero + floor(ival(2)/si) + addone;
IV= [-len_sa+1:0]'*ones(1,nMarkers) + ones(len_sa,1)*pos_end;

complete= find(all(IV>=1 & IV<=size(cnt.x,1),1));
if length(complete)<nMarkers,
  IV= IV(:,complete);
  mrk= mrk_selectEvents(mrk, complete);
  warning('%d segments dropped', nMarkers-length(complete));
  nMarkers= length(complete);
end

epo= struct('fs', cnt.fs);
if isequal(opt.CLab, '*'),
  cidx = 1:numel(cnt.clab);
else
  cidx= util_chanind(cnt, opt.CLab);
end
epo.clab= cnt.clab(cidx);
if util_getDataDimension(cnt)==1
  % 1D data
  epo.x= reshape(cnt.x(IV, cidx), [len_sa nMarkers length(cidx)]);
  epo.x= permute(epo.x, [1 3 2]);
else
  % 2D data
  epo.x= reshape(cnt.x(IV,:,cidx), [len_sa nMarkers size(cnt.x,2) length(cidx)]);
  epo.x= permute(epo.x, [1 3 4 2]);  
end
clear IV

timeival= si*(core_ival + [1 addone]);
%timeival= round(10000*timeival)/10000;
epo.t= linspace(timeival(1), timeival(2), len_sa);

basic_fields_of_mrk= {'y','className','~event'};
fields_to_copy_from_mrk= intersect(fieldnames(mrk), basic_fields_of_mrk,'legacy');
epo= struct_copyFields(epo, mrk, basic_fields_of_mrk);

fields_exclude= union({'clab', 'time'}, basic_fields_of_mrk);
fields_to_copy_from_mrk= setdiff(fieldnames(mrk), fields_exclude);
epo.mrk_info= struct_copyFields(mrk, fields_to_copy_from_mrk);

fields_to_copy_from_cnt= setdiff(fieldnames(cnt), {'x','clab','fs'});
epo.cnt_info= struct_copyFields(cnt, fields_to_copy_from_cnt);
