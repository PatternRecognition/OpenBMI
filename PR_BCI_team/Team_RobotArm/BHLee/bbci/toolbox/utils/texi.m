function str = texi(str);
% TRANSLATES ALL _ to \_

%c = strfind(str,'_');
%for i = length(c):-1:1
%  str = [str(1:c(i)-1),'\' str(c(i):end)];
%end

str= strrep(str, '_', '\_');
