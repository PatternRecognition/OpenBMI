function flag = obj_cmp(a,b);
%compare to variables
c = whos('a');
d = whos('b');

if strcmp(c.class,d.class) & length(c.size)==length(d.size) & all(c.size==d.size) & c.bytes == d.bytes
  switch c.class
   case 'double'
    flag = all(a(:)==b(:));
   case 'char'
    flag = all(a(:)==b(:));
   case 'cell'
    flag = true;
    for i = 1:length(a(:))
      flag = flag*obj_cmp(a{i},b{i});
    end
   case 'struct'
    fi = fieldnames(a);
    fi2 = fieldnames(b);
    flag = obj_cmp(fi,fi2);
    if flag
      for i = 1:length(fi)
        flag = flag*obj_cmp(getfield(a,fi{i}),getfield(b,fi{i}));
      end
    end
  end
else
  flag = false;
end

