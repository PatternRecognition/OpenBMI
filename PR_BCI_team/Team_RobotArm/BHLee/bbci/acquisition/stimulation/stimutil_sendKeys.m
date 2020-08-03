function stimutil_sendKeys(varargin)

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
  'keylist', ['fj'], ...
  'keycatch_position', 'auto');

if isequal(opt.keycatch_position, 'auto'),
  scrpos= get(0, 'ScreenSize');
  h= round(scrpos(4)/3);
  pos= [5 scrpos(4)-43-h scrpos(3)-10 h];
else
  pos= opt.keycatch_position;
end

set(gcf, 'Toolbar','none', ...
    'NumberTitle','off', ...
    'Name','SEND-KEY Figure', ...
    'Position',pos);
  
set(gcf, 'KeyPressFcn', {@stimutil_sendKeys_keyfcn, opt});
return;



function stimutil_sendKeys_keyfcn(obj, evd, opt)

ch= get(gcf,'CurrentCharacter');
if ismember(ch, opt.keylist),
  ppTrigger(ch-'a'+101);
else
  fprintf('non expected character: <%s>\n', ch);
end
