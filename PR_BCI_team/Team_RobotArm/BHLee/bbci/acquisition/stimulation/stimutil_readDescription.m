function desc= stimutil_readDescription(file,varargin)

global BCI_DIR

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'dir',[BCI_DIR 'acquisition/data/task_descriptions/'], ...
                 'suffix','.txt');

desc= textread([opt.dir filesep file opt.suffix],'%s','delimiter','\n');
