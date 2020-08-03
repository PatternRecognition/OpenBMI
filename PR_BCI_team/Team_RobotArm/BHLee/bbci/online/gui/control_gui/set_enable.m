function stat = set_enable(fig,new);

if isfield(get(fig),'Enable')
  stat.field = get(fig,'Enable');
  if ischar(new) 
    set(fig,'Enable',new);
  else
    set(fig,'Enable',new.field);
  end
else
  stat.field = [];
end

ch = get(fig,'Children');

stat.children = cell(1,length(ch));

for i = 1:length(ch)
  if ischar(new)
    stat.children{i} = set_enable(ch(i),new);
  else
    stat.children{i} = set_enable(ch(i),new.children{i});
  end
end