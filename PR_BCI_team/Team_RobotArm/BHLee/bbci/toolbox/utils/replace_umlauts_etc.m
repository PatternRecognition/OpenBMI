function str= replace_umlauts_etc(str)

str= strrep(str, 'ä', 'ae');
str= strrep(str, 'ö', 'oe');
str= strrep(str, 'ü', 'ue');
str= strrep(str, 'Ä', 'Ae');
str= strrep(str, 'Ö', 'Oe');
str= strrep(str, 'Ü', 'Ue');
str= strrep(str, 'ß', 'ss');

str= strrep(str, 'á', 'a');
str= strrep(str, 'à', 'a');
str= strrep(str, 'â', 'a');
str= strrep(str, 'ã', 'a');
str= strrep(str, 'é', 'e');
str= strrep(str, 'è', 'e');
str= strrep(str, 'ê', 'e');
str= strrep(str, 'í', 'i');
str= strrep(str, 'ì', 'i');
str= strrep(str, 'î', 'i');
str= strrep(str, 'ó', 'o');
str= strrep(str, 'ò', 'o');
str= strrep(str, 'ô', 'o');
str= strrep(str, 'õ', 'o');
str= strrep(str, 'ú', 'u');
str= strrep(str, 'ù', 'u');
str= strrep(str, 'û', 'u');
str= strrep(str, 'ñ', 'n');
str= strrep(str, 'ç', 'c');

str= strrep(str, 'Á', 'A');
str= strrep(str, 'À', 'A');
str= strrep(str, 'Â', 'A');
str= strrep(str, 'Ã', 'A');
str= strrep(str, 'É', 'E');
str= strrep(str, 'È', 'E');
str= strrep(str, 'Ê', 'E');
str= strrep(str, 'Í', 'I');
str= strrep(str, 'Ì', 'I');
str= strrep(str, 'Î', 'I');
str= strrep(str, 'Ó', 'O');
str= strrep(str, 'Ò', 'O');
str= strrep(str, 'Ô', 'O');
str= strrep(str, 'Õ', 'O');
str= strrep(str, 'Ú', 'U');
str= strrep(str, 'Ù', 'U');
str= strrep(str, 'Û', 'U');
str= strrep(str, 'Ñ', 'N');
str= strrep(str, 'Ç', 'C');

%% TODO: Ï, Ý, ø, Æ, etc.
%% >> cc= char(64:256); cc(isletter(cc))
