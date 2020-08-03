function str= escape_printf(str)
%ESCAPE_PRINTF - Insert Escape Charaters for Non-Formatted FPRINTF.

str= strrep(str, '\','\\');
str= strrep(str, '%','%%');
