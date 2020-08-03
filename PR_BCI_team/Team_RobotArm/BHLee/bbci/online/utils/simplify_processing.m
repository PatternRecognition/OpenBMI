function [cont_proc,feature,cls] = simplify_processing(cont_proc,feature,cls);
%SIMPLIFY_PROCESSING simplifies a processing for bbci_bet 
%(reduce to non redundant calculations)
%
% usage:
%   [cont_proc,feature,cls] = simplify_processing(cont_proc,feature,cls);
%
% input:
%   cont_proc    a struct array of processings on continuous data
%   feature      a struct array of feature processings
%   cls          a struct array of classifiers
%
% output
%   the same arguments but reduced to non-redundant informations
%
% Description: 
% if cont_proc contains to equal processing this will be shorten to one and the number in feature will be modified.
% the same if true for feature, here cls will be modified
% equal cls will not be reduced
%
% Guido Dornhege, 29/04/05
% $Id: simplify_processing.m,v 1.1 2006/04/27 14:24:59 neuro_cvs Exp $

if nargin<3
  error('not enough input arguments');
end

cind = [];
for i = 2:length(cont_proc)
  for j = 1:i-1
    flag = compare_objects(cont_proc(i),cont_proc(j));
    if flag
      cind = cat(1,cind,[i,j]);
      break;
    end
  end
end


if ~isempty(cind)
  
  poi = 1;
  map = zeros(length(cont_proc),1);
  for i = 1:length(cont_proc)
    ind = find(i==cind(:,1));
    if isempty(ind)
      map(i) = poi;
      poi = poi+1;
    else
      map(i) = map(cind(ind,2));
    end
  end
  cont_proc(cind(:,1)) = [];
  
  for i = 1:length(feature)
    feature(i).cnt = map(feature(i).cnt);
  end
end


cind = [];
for i = 2:length(feature)
  for j = 1:i-1
    flag = compare_objects(feature(i),feature(j));
    if flag
      cind = cat(1,cind,[i,j]);
      break;
    end
  end
end


if ~isempty(cind)
  
  poi = 1;
  map = zeros(length(feature),1);
  for i = 1:length(feature)
    ind = find(i==cind(:,1));
    if isempty(ind)
      map(i) = poi;
      poi = poi+1;
    else
      map(i) = map(cind(ind,2));
    end
  end

  feature(cind(:,1)) = [];
  
  for i = 1:length(cls)
    cls(i).fv = map(cls(i).fv);
  end


end






function flag = compare_objects(a,b);

a1 = whos('a');
b1 = whos('b');

if ~strcmp(a1.class,b1.class) | a1.bytes~=b1.bytes | ndims(a1.size)~=ndims(b1.size) | ~all(a1.size==b1.size)
  flag = false;
  return;
end

ty = a1.class;

switch ty
 case {'logical','double'}
  flag = all(a==b);
 case 'char'
  flag = strcmp(a,b);
 case 'cell'
  a = a(:);
  b = b(:);
  flag = true;
  for i = 1:length(a)
    flag = flag*compare_objects(a{i},b{i});
  end
 case 'struct'
  if prod(size(a))>1
    flag = true;
    a = a(:);
    b = b(:);
    for i = 1:length(a)
      flag = flag*compare_objects(a(i),b(i));
    end
  else
    a1 = sort(fieldnames(a));
    b1 = sort(fieldnames(b));
    flag = compare_objects(a1,b1);
    if flag
      for i = 1:length(a1)
        flag = flag*compare_objects(getfield(a,a1{i}),getfield(b,a1{i}));
      end
    end
  end
end

        