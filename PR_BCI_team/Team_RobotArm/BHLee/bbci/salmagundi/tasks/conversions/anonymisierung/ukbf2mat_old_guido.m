function ukbf2mat(data_in,doc_in,data_out)
% function ukbf2mat(data_in,doc_in,data_out)
%
% data_out: specifies a directory where the matlab and picture
% files are stored. Note: put an '/' at the end! If not specified
% the generated files will be stored in:
% /home/neuro/data/SFB618/data/eeg/eegMat/UKBF/ 
%

if ~isunix  % the program only work on linux platforms
  error('program only supports linux platforms');
end

% parse input
if length(data_in)>=length('.eeg') & strcmp(data_in(end-length('.eeg')+1:end),'.eeg')
  dat_in = data_in;
  data_in = [fileparts(data_in),'/'];
  
  if nargin<2 | isempty(doc_in)
    dd = dir([data_in,'*.doc']);
    if length(dd)==0
      error('no doc file found');
    end
    if length(dd)>1
      error('more than one doc file found');
    end
  
    doc_in = [data_in,dd.name];
  end
  if doc_in(1)~='/'
    doc_in = [data_in,doc_in];
  end
  
else
  if nargin>=3
    error('too many input arguments')
  end
  if nargin==2
    data_out = doc_in;
  end
  
  if data_in(end)~='/'
    data_in = [data_in,'/'];
  end
  dd = dir([data_in,'*.eeg']);
  if length(dd)==0
    error('no EEG data found');
  end
  if length(dd)>1
    error('more than one eeg file found');
  end
  dat_in = [data_in dd.name];
  dd = dir([data_in,'*.doc']);
  if length(dd)==0
    error('no doc file found');
  end
  if length(dd)>1
    error('more than one doc file found');
  end
  
  doc_in = [data_in,dd.name];
end

if ~exist('data_out','var') | isempty(data_out)
  data_out = '/home/neuro/markus/tmp/UKBF/test_parsed_out/';
%  data_out = '/home/neuro/data/SFB618/data/eeg/eegMat/UKBF/';
end

% load the eeg data
[cnt,mrk,mnt] = eeg_conversion(dat_in);

% creating save name
if data_out(end)=='/'
  cnt.info = anonymisierung_doc(doc_in);
  str = cnt.info.ableitung(1:length('xx.xx.xxxx'));
  str = [str(9:10),'_',str(4:5),'_',str(1:2)];
  str2 = cnt.info.eegnr;
  str2(str2=='/') = '_';
  str = [str '_' str2];
  data_out = [data_out str];
end

% load the doc file
cnt.info = anonymisierung_doc(doc_in,[data_out,'.jpg']);

% change title and file name
cnt.file = [data_out];
[aa,bb] = fileparts(data_out);
cnt.title = ['data set ' bb];

% save doc file
save([data_out,'.mat'],'cnt','mrk','mnt');
