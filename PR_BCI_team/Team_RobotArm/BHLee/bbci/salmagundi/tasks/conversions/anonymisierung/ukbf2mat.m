function ukbf2mat(data_in,doc_in,data_out)
% ukbf2mat - function ukbf2mat(data_in,doc_in,data_out)
%
% Synopsis:
%   ukbf2mat(data_in,doc_in,data_out)
%   
% Arguments:
%   data_in: Specifies the eeg data file or the directory where the eeg-
%       and the doc-file can be found. If doc_in is specified, data_in
%       specifies the eeg data file, otherwise, data_in specifies the
%       directory where the doc and the eeg files are stored.
%   doc_in: specifies the doc file
%   data_out: set the output directory, where the mat and picture files
%       are storred. If not specigied, the output files are stored in
%       /home/neuro/data/SFB618/data/eeg/eegMat/UKBF/
%   
% Description:
%   This function makes the eeg- and doc-files of the UKBF
%   EEG-database anonymous. It loads the eeg files, deletes
%   personal information of the patients, converts the data to a
%   matlab format which is readable by standard bci toolbox used at
%   ida using eegfile_loadMatlab function and saves the file to
%   specified location. This matlab function also extracts informations
%   from the doc file and adds it to the matlab file.
%   
% Examples:
%   ukbf2mat('data_dir');
%   ukbf2mat('data_dir/data.eeg','data_dir/data.doc');
%   ukbf2mat('data_dir/data.eeg','data_dir/data.doc','data_out');
%   
% See also: 
% 

% Author(s): Guido Dornhege and Markus Schubert, Oct 2006
% $Id: ukbf2mat.m,v 1.3 2006/10/10 16:01:33 neuro_cvs Exp $

if ~isunix  % the program only work on linux platforms
  error('program only supports linux platforms');
end

% parse input
switch nargin
 case 0
  error('no input arguments specified');
 case 1
  % find doc file in data_in
  if data_in(end) ~= '/'
    data_in = [data_in '/'];
  end
  dd = dir([data_in,'*.doc']);
  switch length(dd)
   case 0
    error('no doc file found');
   case 1
    doc_in = [data_in dd.name];
   otherwise
    error('more than one doc file found');
  end
  % find eeg file in data_in
  dd = dir([data_in,'*.eeg']);
  switch length(dd)
   case 0
    error('no eeg data file found');
   case 1
    data_in = [data_in dd.name];
   otherwise
    error('more than one eeg data file found');
  end
  data_out = '/home/neuro/data/SFB618/data/eeg/eegMat/UKBF/';
 case 2
  data_out = '/home/neuro/data/SFB618/data/eeg/eegMat/UKBF/';
 case 3
  if data_out(end) ~= '/'
    data_out = [data_out '/'];
  end
 otherwise
  error('too many input arguments');
end

% load the eeg data
[cnt,mrk,mnt] = eeg_conversion(data_in);

% parse the doc file and create the picture
cnt.info = anonymisierung_doc(doc_in);

% creating save name
str = cnt.info.ableitung(1:length('xx.xx.xxxx'));
str = [str(9:10),'_',str(4:5),'_',str(1:2)];
str2 = cnt.info.eegnr;
str2(str2=='/') = '_';
str = [str '_' str2];
data_out = [data_out str];

% load the doc file
cnt.info = anonymisierung_doc(doc_in,[data_out,'.jpg']);

% change title and file name
cnt.file = [data_out];
[aa,bb] = fileparts(data_out);
cnt.title = ['data set ' bb];

% save doc file
save([data_out,'.mat'],'cnt','mrk','mnt');
