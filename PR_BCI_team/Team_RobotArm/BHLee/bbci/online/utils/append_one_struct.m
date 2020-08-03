function s = append_one_struct(s,s2);

fi = fieldnames(s2);

for f = 1:length(fi)
  s = set_defaults(s,fi{f},getfield(s2,fi{f}));
end

