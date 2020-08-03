function perform_anonymisation(srcdir,destdir)
% perform_anonymisation - perform_anonymisation(srcdir,destdir)
%
% Synopsis:
%   perform_anonymisation(srcdir,destdir)
%   
% Arguments:
%   srcdir: directory of UKBF data
%   destdir: directory where to save the matlab and picture
%     files. If not specified, it is set to:
%     /home/neuro/data/SFB618/data/eeg/eegMat/UKBF/
%   
% Description:
%   This function performes the anonymisation of UKBF data. It goes
%   recursivly through a directory tree and convertes all files
%   which are found. This functions supports different kinds of
%   data organisation. First: all files are stored in one
%   folder. This implies there is one eeg data file and one doc
%   file for each data record that must have the same filename,
%   except the extension. The other possibility is, that for each
%   data record, there is one directory which contains exactly one
%   doc file and one eeg file. If so, it is assumed, that thay
%   belong together.
%   
%   
% Examples:
%   
%   
% References:
%   
% See also: 
% 

% Author(s): Markus Schubert, Oct 2006
% $Id: perform_anonymisation.m,v 1.1 2006/10/10 16:01:33 neuro_cvs Exp $

% process input
switch nargin
    case 0
        error('no inputs arguments specified');
    case 1
        destdir = '/home/neuro/data/SFB618/data/eeg/eegMat/UKBF/';
    case 2
    otherwise
        error('too many input arguments');
end

if srcdir(end) ~= '/'
  srcdir = [srcdir '/'];
end

dd = dir(srcdir);

% error handling - empty directory
if length(dd) == 2
  return;
end

% is this a leaf dir?
cdir = 0;
for i=3:length(dd)
  cdir = cdir + dd(i).isdir;
end

cdoc = length(dir([srcdir '*.doc'])); % number of doc files in the directory
ceeg = length(dir([srcdir '*.eeg'])); % number of eeg files in the directory

if cdir > 0
  for i=3:length(dd)
    if dd(i).isdir == 1
      % if directory then go into this directory
      perform_anonymisation([srcdir dd(i).name],destdir);
    end
  end
else
  switch cdoc
   case 0
    return;
   case 1
    fdoc = dir([srcdir '*.doc']);
    fdoc = [srcdir fdoc.name];
    feeg = dir([srcdir '*.eeg']);
    feeg = [srcdir feeg.name];
    ukbf2mat(feeg,fdoc,destdir);
   otherwise
    for i=3:length(dd)
      if dd(i).name(end-3:end) == '.doc'
	fdoc = dd(i).name;
	feeg = dir([dd(i).name(end-3:end) '.eeg']);
	if isempty(feeg) == 0
	  ukbf2mat(feeg,fdoc,destdir);
	end
      end
    end
  end
end
