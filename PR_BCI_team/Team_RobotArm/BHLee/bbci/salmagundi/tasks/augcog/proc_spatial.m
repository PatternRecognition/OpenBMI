function [dat,state] = proc_spatial(typ,dat,varargin);

switch typ
    case 'laplace'
        [dat,state] = online_laplace(dat,varargin{:});
    case 'car'
        [dat,state] = online_commonAverageReference(dat,varargin{:});
    case 'cmr'
        [dat,state] = online_commonMedianReference(dat,varargin{:});
end                 

