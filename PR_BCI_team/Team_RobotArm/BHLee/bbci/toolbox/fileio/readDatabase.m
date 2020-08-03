function [expbase, subbase]= readDatabase
%[expbase, subbase]= readDatabase
%
%  OUT  expbase  - database of experiments
%       subbase  - database of subjects

global EEG_CFG_DIR

exp_file= [EEG_CFG_DIR 'experiment_database.txt'];
sub_file= [EEG_CFG_DIR 'subject_database.txt'];

[datestr, subject, paradigm, appendix]= ...
    textread(exp_file, '%s%s%s%s', 'delimiter',',');
datestr= deblank(datestr);
subject= deblank(subject);
paradigm= deblank(paradigm);
appendix= deblank(appendix);
subjectList= textread(sub_file, '%s');

nSubs= length(subjectList);
subbase= struct('name', subjectList, ...
                'code', cellstr([repmat('a', [nSubs 1]) ...
                                 char('a'+ (0:nSubs-1))']));

nExps= length(datestr);
for ie= 1:nExps,
  expbase(ie).file= [subject{ie} '_' datestr{ie} '/' ...
                     paradigm{ie} appendix{ie} subject{ie}];
  expbase(ie).subject= subject{ie};
  is= strmatch(subject{ie}, {subbase.name});
  if ~isempty(is),
    expbase(ie).code= subbase(is).code;
  end
  expbase(ie).paradigm= paradigm{ie};
  expbase(ie).appendix= appendix{ie};
end
