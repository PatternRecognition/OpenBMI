function [cont_proc,feature,cls,post_proc,marker_output] = cmb_setup_docBrowser(bbci,varargin);
% [cont_proc,feature,cls,post_proc,marker_output] = 
%                   cmb_setup_concat(bbci,cls_setup1,cls_setup2,...)
%
% Calculate the concatenation of classifiers as setup for bbci_bet_apply.
% IN:
%  bbci           - runtime variables, loaded from training setup files.
%  cls_setup      - classifier setup, loaded from training setup files.
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

for i = 1:length(varargin),
  cls = [];
  feature = [];
  cont_proc = [];
  post_proc = struct('proc','docBrowserPostProc','proc_param',{});
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
    
    if isempty(cont_proc) & ~isempty(allcont_proc)    
      cont_proc = set_defaults(cont_proc, 'clab', [], 'proc', [], 'proc_param', []); 
    end
    
    if ~isempty(cont_proc) & isempty(allcont_proc)    
      allcont_proc = set_defaults(allcont_proc, 'clab', [], 'proc', [], 'proc_param', []); 
    end
    
    
    allcont_proc = cat(2, allcont_proc, cont_proc);

    if isempty(post_proc) & ~isempty(allpost_proc)
      post_proc = set_defaults(post_proc, 'proc', [], 'proc_param', []);
    end

    if ~isempty(post_proc) & isempty(allpost_proc)
      allpost_proc = set_defaults(allpost_proc, 'proc', [], 'proc_param', []);
    end

    allpost_proc = cat(2, allpost_proc, post_proc);
  
    
    
    if isempty(marker_output) & ~isempty(allmarker_output)
      marker_output = set_defaults(marker_output, 'marker', [], 'value', [], ...
                                             'no_marker', []);
    end
    if isempty(allmarker_output) & ~isempty(marker_output)
      allmarker_output = set_defaults(allmarker_output, 'marker', [], 'value', [], ...
                                             'no_marker', []);
    end
    
    allmarker_output = cat(2, allmarker_output, marker_output);
  end    
end
cls = allcls;
feature = allfeature;
cont_proc = allcont_proc;
post_proc = allpost_proc;
marker_output = allmarker_output;

