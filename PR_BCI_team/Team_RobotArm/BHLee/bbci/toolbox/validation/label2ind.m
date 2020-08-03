function lind= label2ind(label)
%lind= label2ind(label)
%
% IN  label - label array (class affiliation matrix) as field .y in the
%             feature vector struct, usually called fv
% 
% OUT lind  - label indices


%[dmy, lind]= max(label);

%% a bit faster:
nClasses= size(label,1);
lind= [1:nClasses]*label;
