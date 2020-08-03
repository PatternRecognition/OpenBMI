function res = make_proc_apply(com);

persistent app

switch com
 case 'init'
  app = '';
 case 'get'
  if isempty(app)
    app = 'fv=epo;';
  end
  res = app;
 otherwise
  if isempty(app)
    app = com;
  else
    app = sprintf(['%s hlp_epo = epo; epo = fv; ',...
                   '%s epo = hlp_epo;'],app,com);
  end
end
