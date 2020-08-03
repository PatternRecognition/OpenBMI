function Cout= str_rmCommonPostfix(C)
%C= str_rmCommonPostfix(C)

Crev= apply_cellwise(C, 'fliplr');
Crev= str_rmCommonPrefix(Crev);
Cout= apply_cellwise(Crev, 'fliplr');
