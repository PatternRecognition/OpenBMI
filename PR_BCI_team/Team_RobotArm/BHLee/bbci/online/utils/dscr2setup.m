function [cont_proc,feature,cls,post_proc,marker_output] = dscr2setup(dscr);
% CONVERTS OLD ALF INTERFACE SETUP TO BET INTERFACE SETUP

dscr = set_defaults('clab',{{}},'proc_cnt_apply',struct(),'ilen_apply',0,'proc_apply','','model','LDA','C',[],'scale',1);
dscr.proc_cnt_apply = set_defaults('fcn','','param',{{}});

cont_proc = struct('clab',dscr.clab);
cont_proc.procFunc = {dscr.proc_cnt_apply.fcn};
cont_proc.procParam = {dscr.proc_cnt_apply.param};


feature = struct('cnt',1);
feature.ilen_apply = dscr.ilen_apply;
feature.proc = {};
feature.proc_param = {};
c = strfind(dscr.proc_apply,';');
while ~isempty(c);
  str = dscr.proc_apply(1:c(1));
  dscr.proc_apply = dscr.proc_apply(c(1)+1:end);
  d = strfind(str,'=');
  str = str(d(1)+1:end);
  d = strfind(str,'(');
  na = str(1:d(1)-1);
  while na(1)==' '
    na = na(2:end);
  end
  
  feature.proc = cat(2,feature.proc,{na});
  str = str(d(1):end);
  d = strfind(str,',');
  if isempty(d)
    feature.proc_param = cat(2,feature.proc_param,{{}});
  else
    str = str(d(1)+1:end);
    d = strfind(str,')');
    str = str(1:d(end)-1)
    feature.proc_param{end+1} = eval(['{' str '}']);
  end
  c = strfind(dscr.proc_apply,';');
end

cls = struct('fv',1);
cls.applyFcn = getApplyFuncName(dscr.model);
cls.C = dscr.C;


cls.scale = dscr.scale;

  

post_proc = [];
marker_output = [];

