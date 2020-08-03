function dscr = setup2dscr(cont_proc,feature,cls);
% converts bet format to alf


cont_proc = set_defaults(cont_proc,'clab',{{}},'procFunc','','procParam',{{}});
feature = set_defaults(feature,'ilen_apply',0,'proc',{{}},'proc_param',{{{}}});
cls = set_defaults(cls,'applyFcn','apply_separatingHyperplane','C','','scale',1);

dscr = struct('clab',cont_proc.clab);
if length(cont_proc.procFunc)>1
  error('cannot convert');
end

dscr.proc_cnt = struct('fcn',cont_proc.procFunc{:},'param',cont_proc.procParam{:});

dscr.ilen_apply = feature.ilen_apply;

dscr.C = cls.C;
dscr.scale = cls.scale;

c = strfind(cls.applyFcn,'_');
dscr.model = cls.applyFcn(c(1)+1:end);

str = '';

for i = 1:length(feature.proc)
  str2 = sprintf(',%s',feature.proc_param{i});
  str = sprintf('%sfv = %s(fv%s);\n',str,feature.proc{i},str2);
end

dscr.proc_apply = str;
