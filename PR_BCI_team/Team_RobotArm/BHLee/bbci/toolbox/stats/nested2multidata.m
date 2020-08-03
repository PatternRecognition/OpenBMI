function out = nested2multidata(dat, level)
% NESTED2MULTIDATA - convert a data matrix with multiple factors nested in
% the columns into a multidimensional data set.
%
%Usage:
% dat = nested2multidata(dat,level)
%
%Arguments:
% dat      - A 2D N*(F1*F2*...FK)
%             matrix where the factors are nested in the columns. In this
%             case the first factor should be the FASTEST(!!) varying, such
%             that the columns are eg F1_1/F2_1  F1_2/F2_1  F1_1/F2_2  F1_2/F2_2.
%             If you want to use this format, you must specify varnames.
% level    - vector giving the number of levels for each factor eg [2 2].
%
%Returns:
% DAT      -  N*F1*F2*F3*...*FN data matrix. 
%             First dimension (rows): refers to the subjects, ie each row 
%             contains the data of one subject. The second and successive 
%             dimensions contain the
%
% See also rmanova
%
% Note: Un-nested data is necessary for rmanova and plot_stats.
%
% Author(s): matthias treder 2011

out = zeros([size(dat,1) level]);
out(:) = dat(:);
