%
%
% QUALITY ASSESSMENT (QA) METHODS:
% 
% QA methods were written for analyzing data obtained in a psychophysical experiment, in which EEG was recorded concurrently. 
% The majority of the methods are used for plotting (qa_plot*) and classification (qa_cfy*).
%
% The methods require the same data structures as normal bbci-toolbox methods. Additionally, the mrk and epo structs must/might contain
% the following fields:
% 
% .detected   -   a logical vector, indicating whether the stimulus was detected or not
% 
% .stimlev    -   optional; a vector of the same length as field '.className'. Stores the stimulus intensities/levels for each class. 
%                 this field is used for plotting the psychometric function.
% 
%
