function str= replace_umlauts(str)

str= strrep(str, 'ä', 'ae');
str= strrep(str, 'ö', 'oe');
str= strrep(str, 'ü', 'ue');
str= strrep(str, 'Ä', 'Ae');
str= strrep(str, 'Ö', 'Oe');
str= strrep(str, 'Ü', 'Ue');
str= strrep(str, 'ß', 'ss');
