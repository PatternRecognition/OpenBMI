function date_str= getDateTimeString

ve= datevec(now);
v= cat(1, {num2str(ve(1))}, ...
       cellstr(char(max('0',double(num2str(round(ve(2:6))'))))));
date_str= [v{1} sprintf('_%s', v{2:end})];
