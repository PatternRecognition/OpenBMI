function mrk2 = separate_markers(mrk,flag);
%SPLITS markers into separated classes
%
% usage:
%    mrk = separate_markers(mrk,<flag=true>);
%    
% input:
%    mrk:   a usual mrk structure
%    flag:  true: consecutive blocks with same class number were treated as one block, false: otherwise. 
%
% output:
%    mrk.y  is the identity matrix
%    mrk.className is mrk.className + added latin number n for the nth example of this class
%
% Guido Dornhege, 27/04/2004

if ~exist('flag','var') | isempty(flag)
  flag = true;
end

if flag
  mrk2 = copyStruct(mrk,'y','className');
  mrk2.y = [];
  mrk2.className = {};
  lab = 0;
  count = ones(1,size(mrk.y,1));
  for i = 1:size(mrk.y,2);
    ind = find(mrk.y(:,i));
    if ind == lab
      mrk2.y = cat(2,mrk2.y,[zeros(size(mrk2.y,1)-1,1);1]);
    else
      mrk2.y = cat(1,mrk2.y,zeros(1,size(mrk2.y,2)));
      mrk2.y = cat(2,mrk2.y,[zeros(size(mrk2.y,1)-1,1);1]);
      lat = latin(count(ind));
      mrk2.className = {mrk2.className{:},[mrk.className{ind},' ',lat{1}]};
      lab = ind;
      count(ind)=count(ind)+1;
    end
  end
    
else
  n = size(mrk.y,2);
  mrk2 = copyStruct(mrk,'y','className');
  mrk2.className  = cell(1,n);
  mrk2.y = eye(n,n);

  count = ones(1,size(mrk.y,1));

  for i = 1:n
    cl = find(mrk.y(:,i));
    lat = latin(count(cl));
    mrk2.className{i} = [mrk.className{cl},' ',lat{1}];
    count(cl) = count(cl)+1;
  end
end


  