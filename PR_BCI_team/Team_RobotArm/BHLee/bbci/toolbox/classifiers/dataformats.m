function dataformats()
% DATAFORMATS - Description of IDA-Toolbox data formats
%
%   This dummy function describes some basic data formats shared across all
%   functions in the IDA Toolbox. In particular, it describes the data
%   format required for classifiers and for processing utilities. Also, the
%   general scheme for passing options is explained.
%
%   DATA FOR CLASSIFIERS:
%
%   Functions for training a classifier typically require input points
%   and labels as separate input variables.
%   The input points are passed as a 2-dimensional matrix X, where each
%   column of X corresponds to one data point. X thus has dimension 
%   [NDIMS NPOINTS] for NPOINTS data points, with each data point having
%   NDIMS dimensions.
%
%   Labels are passed as a matrix containing 0 and 1. For an M-class
%   problem, LABELS is a matrix of size [M NPOINTS], where the i.th column
%   of LABELS contains the labels for the i.th data point. LABELS(j,i)==1 if
%   point X(:,i) is a member of class j. Mind that this format allows
%   each point to belong to more than one class.
%
%   DATA FOR PROCESSING UTILITIES:
%
%   Most processing routines accept data in a combined format, where
%   input data and labels are merged together into a structure. This
%   structure, most often called FV (for 'Feature Vectors') in the
%   routines, has two fields, FV.x (input data) and FV.y (labels).
%
%   FV.x are the input points as a matrix. For typical vector data, FV.x
%   has size [NDIMS NPOINTS] for NPOINTS input points, each point being
%   an NDIMS-dimensional vector. For data stemming from multi-variate
%   time-series, FV.x is a 3-dimensional matrix of size 
%   [NTIMESTEPS NDIMS NPOINTS]. This corresponds to NPOINTS samples from
%   a NDIMS-variate time-series, where each time series is measured at
%   NTIMESTEPS points.
%   
%   FV.y are the labes, given as a matrix of 0 and 1. The format is the
%   same as described above for classifiers, so that FV.y(j,i)==1 if the
%   the i.th data point is member of class j.
%
%   PASSING OPTIONS
%   
%   Optional arguments are passed as parameter/value pairs, similar to
%   the SET/GET commands in Matlab. A function FUN requiring one
%   mandatory argument ARG could have its options passed in several
%   variants:
%   FUN(ARG, 'OptionName', value) sets one option, indentified by the
%   string 'OptionName', to the given value.
%   FUN(ARG, 'Option1', value1, 'Option2', value2, ...) can set several
%   options at once.
%   FUN(ARG, OPT) where OPT is a structure returned by
%   PROPERTYLIST2STRUCT. This structure contains a field for each option,
%   with its corresponding value.
%   FUN(ARG, OPT, 'option1', value1) uses the options given in structure
%   OPT, but overrides the option 'option1' with value1.
%   
%   See also PROPERTYLIST2STRUCT,ISPROPERTYSTRUCT
%

% Copyright Fraunhofer FIRST.IDA (2004)
% $Id: dataformats.m,v 1.1 2004/08/16 11:49:53 neuro_toolbox Exp $
