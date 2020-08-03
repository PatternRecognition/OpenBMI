function s = setfields(s,var,val);

str = sprintf('s.%s = val;',var);
eval(str);
