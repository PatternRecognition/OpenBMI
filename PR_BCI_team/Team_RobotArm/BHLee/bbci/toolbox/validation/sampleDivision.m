function [divTr, divTe, nPick]= sampleDivision(g, nDivisions, nPick, skew)
%[divTr, divTe]= sampleDivision(goal, nDivisions, nPick, skew)
%
% skew (optional): If specified it must contain for each class a factor
% by which the final number of samples in the training set from that
% class is scaled. If this factor is smaller than 1 samples are left out
% at random. If it is larger than 1, the original training set is
% extended by a random sub-sample (with rep.) from itself, such that the
% final divTr has skew(cl) elements from that class. For skew(cl) == 1
% nothing special is done.
%
% g= randn(1,100)>0;
% g(2,:)= 1-g;
% [divTr, divTe]= sampleDivision(g, 5);

  
if nargin < 4
  skew = [];
end

nClasses= size(g,1);
nEventsInClass= sum(g,2);

if ~exist('nPick','var'),
  nPick= nEventsInClass;
end

if nDivisions==1,                          %% leave-one-out
%  msg= 'for nDivisions==1 we do leave-one-out';
%  bbci_warning(msg, 'validation', mfilename);
  idx= [];
  for cl= 1:nClasses,
    ci= find(g(cl,:));
    idxCl= randperm(nEventsInClass(cl));
    idx= [idx ci(idxCl(1:nPick(cl)))];
  end
  totalPick= sum(nPick);
  divTe= num2cell(idx',2);
  divTr= idx' * ones(1,totalPick,1);
  divTr(1:totalPick+1:end)= [];
  divTr= num2cell(reshape(divTr', totalPick, totalPick-1), 2);
%  divTr= cell(totalPick, 1);              %% loopy version
%  for k= 1:totalPick,
%    divTr{k}= idx(setdiff(1:totalPick, k));
%  end

else                                       %% cross-validation
  divTr= cell(nDivisions, 1);
  divTe= cell(nDivisions, 1);

  for cl= 1:nClasses,
    ci= find(g(cl,:));
    idx= randperm(nEventsInClass(cl));
    div= round(linspace(0, nPick(cl), nDivisions+1));
    for d= 1:nDivisions,
      sec= ci(idx(div(d)+1:div(d+1)));
      for dd= 1:nDivisions,
        if dd==d,
          divTe{dd}= [divTe{dd} sec];
        else
          divTr{dd}= [divTr{dd} sec];
        end
      end
    end
  end
 
  % If necessary skew the distributions in the training sets by up- or
  % downsampling it.
  if ~isempty(skew)
    for d=1:nDivisions
      dTr = divTr{d};
      nTr = length(divTr{d});
      for cl=1:nClasses
	if skew(cl) == 1
	  % Do nothing
	elseif skew(cl) < 1
	  % Take random subset of training set
	  idx1= find(g(cl,dTr)); % Target class
	  idx2= find(~g(cl,dTr)); % Others
	  idx1= idx1(randperm(length(idx1))); % Permute 
	  idx1= idx1(1:round(nTr*skew(cl))); % Cut off
	  dTr= [dTr(idx1), dTr(idx2)]; % Rebuild indices
	else % skew > 1
	  idx1 = find(g(cl,dTr)); % Target class
	  idx2 = find(~g(cl,dTr)); % Others
	  idx3 = ceil(length(idx1)*rand(round(skew(cl)-1)*length(idx1),1));
	        % Resample with repetition
	  idx1 = [idx1, idx1(idx3)]; % Original set plus new elements
	  dTr= [dTr(idx1), dTr(idx2)]; % Rebuild indices
	end
      end
      divTr{d} = dTr;
    end
  end
end


for dd= 1:nDivisions,
  divTr{dd}= divTr{dd}(randperm(length(divTr{dd})));
  divTe{dd}= divTe{dd}(randperm(length(divTe{dd})));
end
