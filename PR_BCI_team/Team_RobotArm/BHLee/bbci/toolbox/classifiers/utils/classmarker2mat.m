function [y, classname]= classmarker2mat(z)

label_set= unique(z);
nClasses= length(label_set);
nSamples= length(z);

if iscell(label_set),
  classname= label_set;
else
  classname= cellstr([ repmat('class ',nClasses,1) num2str(label_set') ]);
  z= num2cell(z);
end

y= zeros(nClasses, nSamples);
for ii= 1:nClasses,
  if ischar(z{1}),
    idx= strmatch(label_set{ii}, z, 'exact');
  else
    idx= find(label_set(ii)==z);
  end
  y(ii,idx)= 1;
end
