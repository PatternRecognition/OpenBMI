function bbci_bet_bugreport(subject,string);
%BBCI_BET_BUGREPORT adds string to the bugreport list
% 
% usage:
%   bbci_bet_bugreport(string);
%
% input:
%   subject :  the subject who submits the bug
%   string  :  a string which describes the bug
% 
% The file BCI_DIR/bbci_bet/bugliste/general_buglist.txt will be extended by this string and committed to the CVS
%
% Guido Dornhege, 04/05/05
% $Id: bbci_bet_bugreport.m,v 1.1 2006/04/27 14:24:59 neuro_cvs Exp $

global BCI_DIR

if isempty(string)
  return;
end

if isunix
  str = sprintf('cd %sbbci_bet/bugliste/; cvs update general_buglist.txt;',BCI_DIR);
else
  str = sprintf('%s\\bbci_bet\\utils\\update_bugliste',BCI_DIR(1:end-1));
end
system(str);



fid = fopen([BCI_DIR 'bbci_bet/bugliste/general_buglist.txt'],'a');

fprintf(fid,'\n\n\n');
fprintf(fid,'%s has submitted a bug on %s:\n\n',subject,datestr(now));
fprintf(fid,'%s\n\n',string);
fclose(fid);

if isunix
  str = sprintf('cd %sbbci_bet/bugliste/; cvs commit -m "%s has submitted a bug" general_buglist.txt;',BCI_DIR,subject);
else
  str = sprintf('%s\\bbci_bet\\utils\\commit_bugliste %s',BCI_DIR(1:end-1),subject);
end

system(str);