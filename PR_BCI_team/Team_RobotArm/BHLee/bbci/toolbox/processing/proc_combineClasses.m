function epo = proc_combineClasses(epo,varargin)
%PROC_COMBINECLASSES combines Classes but gives the opportunity to
%remember the old label structure. THe programm is needed for
%train_combineClasses, due to the fact that sometimes it could be
%of high interest to train a classifier in multiclass but then
%combine classes to one.
%
% usage:
%      epo = proc_combineClasses(epo,classes,....);
%
% input:
%      epo        a usual epo structure
%      classes    is a cell array of classes (out of epo.className)
%                 or a number and describes which classes belong
%                 together
%                 it can be a char or number (then this class is alone)
%                 if all classes in the call of the function are
%                 chars or number it is handled as one cell array,
%                 this means the classes are combined together.
%                 All classes which were not called are directly
%                 passed to the output structure without any
%                 changes.
%      
% output:
%      epo       a epo structure with combined classes where
%                different integers are chosen in each class to
%                divide the combination. Classname are combined by &
%
% example: (let 'left','right','foot' the classNames)
%    epo = proc_combineClasses(epo,'left',{'right','foot'});
%         will give an epo with one class left and one class as
%         combination of right and foot, the same is done by
%    epo = proc_combineClasses(epo,{'right','foot'});
%         and by
%    epo = proc_combineClasses(epo,'right','foot');
%         or (in classNumbers:)
%    epo = proc_combineClasses(epo,2,3);
%
% GUIDO DORNHEGE, 02/04/03

flag = 0;
nClasses = size(epo.y,1);
clInd = zeros(2,nClasses);   %1st row place for each class, 2nd row
                             %number in this class
row = 1;
for i = 1:length(varargin)
  var = varargin{i};
  if isnumeric(var)
    clInd(1,var) = row;
    clInd(2,var) = 1;
  elseif ischar(var)
    var = find(strcmp(epo.className,var));
    clInd(1,var) = row;
    clInd(2,var) = 1;
  else
    if length(var)==1
      var = var{1};
      if isnumeric(var)
	clInd(1,var) = row;
	clInd(2,var) = 1;
      else 
	var = find(strcmp(epo.className,var));
	clInd(1,var) = row;
	clInd(2,var) = 1;
      end
    else
      flag = 1;
      for j = 1:length(var)
	vara = var{j};
	if isnumeric(vara)
	  clInd(1,vara) = row;
	  clInd(2,vara) = j;
	else 
	  vara = find(strcmp(epo.className,vara));
	  clInd(1,vara) = row;
	  clInd(2,vara) = j;
	end
      end
    end
  end
  row = row+1;
end

    
if flag==0
  ind = find(clInd(1,:));
  clInd(1,ind) = 1;
  clInd(2,ind) = 1:length(ind);
  row = 2;
end

ind = find(clInd(1,:)==0);

for i = ind;
  clInd(1,i) = row;
  row = row+1;
  clInd(2,i) = 1;
end

row = row-1;
y = zeros(row,size(epo.y,2));
for i = 1:nClasses
  y(clInd(1,i),find(epo.y(i,:))) = clInd(2,i);
end
epo.y = y;

className = cell(1,row);
for i = 1:row
  ind = find(clInd(1,:)==i);
  className{i} = epo.className{ind(find(clInd(2,ind)==1))};
  for j = 2:length(ind)
    className{i} = sprintf('%s & %s',className{i}, ...
			   epo.className{ind(find(clInd(2,ind)==j))});
  end
end

epo.className = className;
