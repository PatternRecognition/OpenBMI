function [out_param str_function in_dat in_param] = opt_funcParsing( str )
%OPT_FUNCPASING Summary of this function goes here
%   Detailed explanation goes here
str2=opt_getToken(str, '=')
str_out=str2{1};
str_out=strrep(str_out,'[','');
str_out=strrep(str_out,']','');
out_param=opt_getToken(str_out,',');

str_in=str2{2,:} %input paremters
str_in2=opt_getToken(str_in,'(')
str_function=str_in2{1}
str_param=str_in2{2,:}

str_param=opt_getToken(str_param, '{');
str_param_dat=str_param{1,:}
in_dat=opt_getToken(str_param_dat, ',');
str_param_pa=opt_getToken(str_param{2,:},';')

str_param_pa{end}=strrep(str_param_pa{end}, ')', ''); %마지막 부분 ) 제거
str_param_pa{end}=strrep(str_param_pa{end}, '}', '');

for i=1:length(str_param_pa)
    tm=opt_getToken(str_param_pa{i},',');
    if ~isempty(strfind(tm{2},'"')) %string
        tCHAR='string'
    elseif ~isempty(strfind(tm{2},'['))  % number
        tCHAR='number'
    else % variable
        tCHAR='variable'
    end
    for j=1:length(tm)
        tm{j}=strrep(tm{j},'"',''); % 숫자로 들어올대 예외 처리 추가할것
    end
    switch tCHAR
        case 'string'
            in_param.(tm{1})=tm{2};
        case 'number'
            tm{2}=strrep(tm{2},'[','');
            tm{2}=strrep(tm{2},']','');
            in_param.(tm{1})=str2num(tm{2}); % 실수만 받아옴, double은 두개이상일때 x
        case 'variable'
            warning('variable can not be assigned')
    end
    
end


end

