function expbase= expbase_read(varargin)

global EEG_CFG_DIR

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'file_name', [EEG_CFG_DIR 'experiment_database.txt']);

[d,s,p,a]= textread(opt.file_name, '%s%s%s%s', 'delimiter',',');
expbase= struct('subject',s, 'date',d, 'paradigm',p, 'appendix',a)';
