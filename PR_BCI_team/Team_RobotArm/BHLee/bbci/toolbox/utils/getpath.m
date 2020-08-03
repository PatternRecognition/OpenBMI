function p = getpath(g);

if isunix
    c = [0,strfind(g,':')];
else
    c = [0,strfind(g,';')];
end
	
p = cell(length(c)-1,1);

for i = 2:length(c)
    p{i-1}=g(c(i-1)+1:c(i)-1);
end
	
