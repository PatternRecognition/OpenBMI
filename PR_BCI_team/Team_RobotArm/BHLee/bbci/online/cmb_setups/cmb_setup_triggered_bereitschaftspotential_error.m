function [cont_proc,feature,cls,post_proc,marker_output] = cmb_setup_triggered_bereitschaftspotential_error(bbci,varargin);

for i = 1:length(varargin),
  cls = [];
  feature = [];
  cont_proc = [];
  post_proc = [];
  marker_output = [];

  % loads cont_proc, feature, cls, post_proc,marker_output if provided
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
    allcls = cls;
    allfeature = feature;
    allcont_proc = cont_proc;
    allpost_proc = post_proc;
    allmarker_output = marker_output;
  else
    % feature indices need to go into a concatened feature vector, but
    % the individual classifiers don't know that. so we need to add the indices
    cls.fv = cls.fv+lenght(allfeature);
    feature.cnt = feature.cnt+lenght(allcont_proc);
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

if ~isfield(bbci,'triggerMarker') | isempty(bbci.triggerMarker)
  bbci.triggerMarker = [bbci.classDef{1,:}];
end

if ~isfield(bbci,'timeshift') | isempty(bbci.timeshift)
  bbci.timeshift = 0;
end

cm = {};
for i = 1:length(cls)-1
  if iscell(bbci.triggerMarker)
    str = '';
    for j = 1:length(bbci.triggerMarker)
      if isnumeric(bbci.triggerMarker{j})
        str = sprintf('%s%d,',str,bbci.triggerMarker{j});
      else
        str = sprintf('%s''%s'',',str,bbci.triggerMarker{j});
      end
    end
    cm = cat(2,cm,bbci.triggerMarker);
  else
    str = sprintf('%s,',bbci.triggerMarker);
    cm = cat(2,cm,num2cell(bbci.triggerMarker));
  end
  cls{i}.condition = sprintf('M({{%s},[%g,%g]});',str(1:end-1),bbci.timeshift,bbci.timeshift);
end

if ~isfield(bbci,'errorTrigger') | isempty(bbci.errorTrigger)
  bbci.errorTrigger = cm;
end

if ~isfield(bbci,'errorJit') | isempty(bbci.errorJit)
  bbci.errorJit = feature(end).ilen_apply;
end

if iscell(bbci.errorTrigger)
  str = '';
  for j = 1:length(bbci.errorTrigger)
    if isnumeric(bbci.errorTrigger{j})
      str = sprintf('%s%d,',str,bbci.errorTrigger{j});
    else
      str = sprintf('%s''%s'',',str,bbci.errorTrigger{j});
    end
  end
else
  str = sprintf('%s,',bbci.errorTrigger);
end
cls{end}.condition = sprintf('M({{%s},[%g,%g]});',str(1:end-1),bbci.errorJit);

