function s = getfields(s,var);

str = sprintf('s = s.%s;',var);
eval(str);
