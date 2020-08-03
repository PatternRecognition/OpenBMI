function expbase= expbase_generate(varargin)
%expbase= expbase_generate(opt)
%
% returns a struct with field
%  .subject  - name of the subject
%  .date     - date of the expriment as string
%  .paradigm - general type of experiment
%  .appendix - specifying the expriment
%
%
% IN: opt - struct and/or propertylist of options with fields
%      .data_dir  - default global EEG_RAW_DIR
%      .extension - default 'vhdr'
%      .save      - save database into a text file, default 0
%      .file_name - default [EEG_CFG_DIR 'experiment_database.txt']

%% bb ida.first.fhg.de 07/2004

global EEG_RAW_DIR EEG_CFG_DIR

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'data_dir', EEG_RAW_DIR, ...
                  'extension', 'vhdr', ...
                  'save', 0, ...
                  'file_name', [EEG_CFG_DIR 'experiment_database.txt']);

d= dir(opt.data_dir);
d= d(3:end);

kk= 0;
for ii= 1:length(d),
  sub_dir= d(ii).name;
  iu= min(find(sub_dir=='_'));
  date_str= sub_dir(iu+1:end);
  if length(date_str)~=8,
    continue;
  end
  dd= dir([opt.data_dir '/' sub_dir]);
  file_list= {dd.name};
  idx= strpatternmatch(['*.' opt.extension], file_list);
  uninteresting= strmatch('impedances', file_list);
  uninteresting= union(uninteresting, strmatch('calibration', file_list));
  idx= setdiff(idx, uninteresting);
  for jj= idx,
    kk= kk+1;
    [subject, date_str, paradigm, appendix]= ...
        expbase_decomposeFilename(dd(jj).name, sub_dir);
    expbase(kk)= struct('subject',subject, 'date',date_str, ...
                        'paradigm',paradigm, 'appendix',appendix);
  end
end
[so,si]= sort({expbase.date});
expbase= expbase(si);

if opt.save,
  fid= fopen(opt.file_name, 'w');
  for ii= 1:length(expbase),
    fprintf(fid, '%s, %s, %s, %s\n', expbase(ii).date, expbase(ii).subject, ...
            expbase(ii).paradigm, expbase(ii).appendix);
  end
  fclose(fid);
end
