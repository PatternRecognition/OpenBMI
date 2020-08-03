function bbci = bbci_apply_convertSettings(settings),
%BBCI_APPLY_CONVERTSETTINGS - Convert old settings format to the new one
%
%Synopsis:
%  BBCI= bbci_apply_convertSettings(SETTINGS)
%
%Arguments:
%  SETTINGS - Old bbci setup struct for bbci_bet_apply.
%
%Output:
%  BBCI - New bbci setup struct for bbci_apply.
%
%WARNING: There is a problem for marker-based classification (as for 
%  ERP spellers, that has to be corrected manually. Ask Benjamin if you
%  need to do this. Anyway, this function should be superfluous soon.

% 02-2011 Martijn Schreuder
% 11-2011 Benjamin: set control.condition.overrun and added warning

global acquire_func general_port_fields

bbci = struct();

% set source
if ~isempty(acquire_func), bbci.source(1).acquire_fcn = acquire_func; end;
bbci.source(1).min_blocklength = settings.bbci.minDataLength;

% set marker. Use defaults, so nothing to set

% set signal
for i = 1:length(settings.cont_proc),
    bbci.signal(i).source = 1;
    if isfield(settings.cont_proc(i), 'procFunc') & ~isempty(settings.cont_proc(i).procFunc),
        for j = 1:length(settings.cont_proc(i).procFunc),
            bbci.signal(i).fcn{j} = str2func(settings.cont_proc(i).procFunc{j});
            bbci.signal(i).param{j} = settings.cont_proc(i).procParam{j};
        end
    end
    bbci.signal(i).clab = settings.cont_proc(i).clab;
end

% match cls with feature
clsToFeat = [settings.cls(:).fv];

% set feature
overrun= -inf;
for i = 1:length(settings.feature),
    bbci.feature(i).signal = settings.feature(i).cnt;
    mtchCls = find(clsToFeat == i,1); % only one ival can be set, taking first
    if isfield(settings.cls(mtchCls), 'condition') & ~isempty(settings.cls(mtchCls).condition),
       i1= find(settings.cls.condition=='[');
       i2= find(settings.cls.condition==' ');
       ival_end= eval(settings.cls.condition(i1+1:i2));
       bbci.feature(i).ival = ival_end + [-settings.feature.ilen_apply 0];
       overrun= max(overrun, ival_end);
    else
        bbci.feature(i).ival = [-settings.feature(i).ilen_apply 0];
    end
    for j = 1:length(settings.feature(i).proc),
        bbci.feature(i).fcn{j} = str2func(settings.feature(i).proc{j});
        bbci.feature(i).param{j} = settings.feature(i).proc_param{j};
    end
end


% set classifier
for i = 1:length(settings.cls),
    bbci.classifier(i).feature = settings.cls(i).fv;
    bbci.classifier(i).apply_fcn = str2func(settings.cls(i).applyFcn);
    bbci.classifier(i).C = settings.cls(i).C;
end

% set control
fb_func = '';
if isfield(settings.bbci, 'feedback') && ~isempty(settings.bbci.feedback), 
  fb_func= str2func(['bbci_bet_feedbacks_' settings.bbci.feedback]); 
end

for i = 1:length(bbci.classifier),
    bbci.control(i).classifier = i;
    bbci.control(i).fcn = fb_func;
    bbci.control(i).param = {};
    if isfield(settings.cls(i), 'condition') & ~isempty(settings.cls(i).condition),
        bbci.control(i).condition.marker = condition_to_marker(settings.cls(i).condition);
        if overrun>0,
          % if there are multiple features and controls, this would need to
          % be changed.
          bbci.control(i).condition.overrun = overrun;
        end
    end
end

% set feedback
if isfield(general_port_fields, 'feedback_receiver'),
    for i = 1:length(bbci.control),
        bbci.feedback(i).control = i;
        bbci.feedback(i).receiver = general_port_fields.feedback_receiver;
    end
end

% set adaptation. Simply copy the info
if isfield(settings.bbci, 'adaptation'),
    bbci.adaptation.active = settings.bbci.adaptation.running;
    bbci.adaptation.fcn = str2func(['bbci_adaptation_' settings.bbci.adaptation.policy]);
    bbci.adaptation.param = {settings.bbci.adaptation(:)};
end

% set quit_condition. Only standard settings

% set log. Only standard settings
if settings.bbci.log,
    bbci.log.output = 'file';
    if isfield(settings.bbci, 'logfilebase'),
        bbci.log.filebase = settings.bbci.logfilebase;
    end
else
    bbci.log.output = 0;
end

end

function marker = condition_to_marker(condition),
    st_pos = find(condition == '{');
    en_pos = find(condition == '}');
    marker = str2num(condition(st_pos(end)+1:en_pos(1)-1));
end
