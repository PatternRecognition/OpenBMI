function [cont_proc,feature,cls,post_proc,marker_output] = cmb_setup_detected_bereitschaftspotential(bbci,varargin);
% [cont_proc,feature,cls,post_proc,marker_output] = 
%     cmb_setup_detected_bereitschaftspotential(bbci,cls_setup1,cls_setup2)
%
% Calculate the concatenation of classifiers as setup for bbci_bet_apply.
% IN:
%  bbci           - runtime variables, loaded from training setup files.
%  cls_setup      - classifier setup, loaded from training setup files.
%                   It is assumed that the dtct setup precedes the dscr setup.
% OUT:
%  cont_proc      - options for continuous data processing
%  feature        - options for feature extraction
%  cls            - options for classifiers
%  post_proc      - options for postprocessing of classifier outputs
%  marker_output  - options for handling the marker information.
%
% See bbci_bet_apply for a detailed specification of these parameters.
%

% dornheg,kraulem 07/05

if length(varargin)~=2
  error('Exactly 2 setup files required');
end

for i = 1:length(varargin),
  cls = [];
  feature = [];
  cont_proc = [];
  post_proc = [];
  marker_output = [];

  % loads bbci, cont_proc, feature, cls, post_proc,marker_output if provided
  load(varargin{i},'cont_proc', 'feature', 'cls', 'post_proc','marker_output');
  
  
  % We have a slight problem here: the individual finish routines do not
  % always generate all fields required for bbci_bet_apply.
  cls = set_defaults(cls, 'condition', [], 'conditionParam', [], 'fv', ...
                          [], 'applyFcn', [], 'C', [], 'integrate', [], ...
                          'bias', [], 'scale', [], 'dist', [], 'alpha', ...
                          [], 'range', [], 'timeshift', []);
  feature = set_defaults(feature, 'cnt', [], 'ilen_apply', [], 'proc', ...
                                  [], 'proc_param', []);
  cont_proc = set_defaults(cont_proc, 'clab', [], 'proc', [], 'proc_param', []);
  post_proc = set_defaults(post_proc, 'proc', [], 'proc_param', []);
  marker_output = set_defaults(marker_output, 'marker', [], 'value', [], ...
                                             'no_marker', []);
  if i==1,
    %initiate allcls
    allcls = cls;
    allfeature = feature;
    allcont_proc = cont_proc;
    allpost_proc = post_proc;
    allmarker_output = marker_output;
  else
    % concatenate cls to allcls.
    % 
    % feature indices need to go into a concatened feature vector, but
    % the individual classifiers don't know that. so we need to add the indices
    cls.fv = cls.fv+length(allfeature);
    feature.cnt = feature.cnt+length(allcont_proc);
    allcls = cat(2, allcls, cls);
    allfeature = cat(2, allfeature, feature);
    allcont_proc = cat(2, allcont_proc, cont_proc);
    allpost_proc = cat(2, allpost_proc, post_proc);
    allmarker_output = cat(2, allmarker_output, marker_output);
  end    
end
cls = allcls;
feature = allfeature;
cont_proc = allcont_proc;
post_proc = allpost_proc;
marker_output = allmarker_output;


if ~isfield(bbci,'detection_triggerMarker') | isempty(bbci.detection_triggerMarker)
  % do dtct-classification on every marker encountered; given
  % bbci_detection_ival is not empty.
  bbci.detection_triggerMarker = [bbci.classDef{1,:}];
end

if ~isfield(bbci,'detection_ival') 
  % always do dtct-classification.
  bbci.detection_ival = [];
end

if ~isempty(bbci.detection_ival)
  % A condition must be formed, restricting the classification of dtct.
  if iscell(bbci.detection_triggerMarker)
    str = '';
    for j = 1:length(bbci.detection_triggerMarker)
      if isnumeric(bbci.detection_triggerMarker{j})
        str = sprintf('%s%d,',str,bbci.detection_triggerMarker{j});
      else
        str = sprintf('%s''%s'',',str,bbci.detection_triggerMarker{j});
      end
    end
  else
    str = sprintf('%s,',bbci.detection_triggerMarker);
  end
  if length(bbci.detection_ival)==1
    bbci.detection_ival = [1 1]*bbci.detection_ival;
  end
  cls{1}.condition = sprintf('M({{%s},[%g,%g]});',str(1:end-1),bbci.detection_ival(1),bbci.detection_ival(2));
end

if ~isfield(bbci,'restrictDiscrimination')
  bbci.restrictDiscrimination = true;
end

if bbci.restrictDiscrimination
  cls{2}.condition = 'F(cl{1}>0);';
end
