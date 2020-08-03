function label = ind2label(ind)
% ind21abel  - convert the index to a label representation
%
% Synopsis:
%   label = ind2label(ind)
%
% Arguments:
%   ind    - a vector of class indices in the range from 1 to n (with
%            possibly missing values).
%
% Returns:
%   label  - a label matrix as expected by the fv structure (i.e. with 1
%            in the position i,j if example j belongs to class i, and 0
%            otherwise)
%
% See also:
%   label2ind
%
% $Id: ind2label.m,v 1.1 2005/02/22 10:28:12 neuro_toolbox Exp $
%
% Copyright (C) 2005 Fraunhofer FIRST
% Author: Pavel Laskov (laskov@first.fhg.de)

[classes,I,J] = unique(ind);
k = length(ind);
d = max(classes);
idx = 1:length(ind);
label = zeros(d,k);

ii = (idx-1)*d + ind;
label(ii) = 1;
