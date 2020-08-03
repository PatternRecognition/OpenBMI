function exper = readFiles(subject, paradigm, appendix)
% function exper = readFiles(subject, paradigm, appendix)
%
% find all files for the given subject and paradigm.
% wildcard '*' at the end of an entry matches all items starting with this string.
% Example: exper = readFiles('*',{'imag_*','self*'},'*')
%          finds all files with imag_curs/lett/move etc. and selfpaced studies.

% kraulem 09/07/2004

[dat, sub] = readDatabase;

if ~iscell(subject)
  subject = {subject};
end
if ~iscell(paradigm)
  paradigm = {paradigm};
end
if nargin<3
  appendix = '';
end
if ~iscell(appendix)
  appendix = {appendix};
end
exper = [];
for i = 1:length(subject)
  for j = 1:length(paradigm)
    for k = 1:length(appendix)
      ind = findstr(subject{i},'*');
      if ~isempty(ind)
	sub = subject{i}(1:(ind(1)-1));
	newfiles = dat(strmatch(sub,{dat.subject}));
      else
	sub = subject{i};
	newfiles = dat(find(strcmp(subject{i},{dat.subject})));
      end
      ind = findstr(paradigm{j},'*');
      if ~isempty(ind)
	par = paradigm{j}(1:(ind(1)-1));
	newfiles = newfiles(strmatch(par,{newfiles.paradigm}));
      else
	par = paradigm{j};
	newfiles = newfiles(find(strcmp(paradigm{j},{newfiles.paradigm})));
      end
     ind = findstr(appendix{k},'*');
      if ~isempty(ind)
	app = appendix{k}(1:(ind(1)-1));
	newfiles = newfiles(strmatch(app,{newfiles.appendix}));
      else
	app = appendix{k};
	newfiles = newfiles(find(strcmp(appendix{k},{newfiles.appendix})));
      end
      exper = [exper, newfiles];
    end
  end      

end


