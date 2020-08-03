function T = generate_tree(opt,written)
% T = generate_tree(opt)
% produce a speller-tree from text file with letters and probabilities.
% opt.file specifies the filename of the text file, relative to BCI_DIR/pseudo_online.

% kraulem 6/7/2004

%if ~isfield(opt,'language')
%  opt.language = 'deAlphaProb.tab';
%end
 
if nargin<=1
    written = '';
end

global BCI_DIR
opt = set_defaults(opt, 'file', 'deAlphaProb.tab',...
                        'language_model',[],...
                        'optimal_partition',1,...
                        'numboxes', 3);
if ~isempty(opt.file)
    [letters,probs] = textread([BCI_DIR 'pseudo_online/' opt.file],'%s%f',-1);
    probs = probs/sum(probs);
    letters = [letters{:}];
end
if ~isempty(opt.language_model)
    lm= lm_loadLanguageModel(opt.language_model);
    probs = lm_getProbability(lm, written, opt);
    if isempty(opt.file)
        letters = lm.charset;
    else
        [letters,dum,ind] = intersect(letters,lm.charset);
        probs = probs(ind);
    end
end

if opt.backspace>0
    letters = ['<' letters];
    probs = [opt.backspace probs'];
    probs = probs/sum(probs);
end
if opt.optimal_partition
    T = convert_tree(optimal_tree(probs),letters);
else
    T = add_subtree([],1,[],letters,probs,opt);
end
return

function T = convert_tree(tr,letters);

T = struct('parent',[],'children',[zeros(1,2)],'leaves',letters);
tr_queue = {{tr.lefttree,tr.left,1,1},{tr.righttree,tr.right,1,2}};

while ~isempty(tr_queue);
    act = tr_queue{1};
    T(end+1).leaves = letters(act{2});
    T(end).parent = act{3};
    T(act{3}).children(act{4}) = length(T);
    if ~isempty(act{1}) 
        tr_queue = {{act{1}.lefttree,act{1}.left,length(T),1},{act{1}.righttree,act{1}.right,length(T),2},tr_queue{2:end}};
    else
        tr_queue = tr_queue(2:end);
    end
end

return;


function tr = optimal_tree(p);

tr = cell(length(p),length(p));

for i = 1:length(p)
  tr{i,i} = {0,p(i),i};
end

for i = 2:length(p)
  for j = 1:length(p)-i+1
    splitn = inf;
    for k = j:j+i-2
      val = tr{j,k}{1} + tr{k+1,j+i-1}{1} + tr{j,k}{2} +  tr{k+1,j+i-1}{2};
      pum = tr{j,k}{2} +  tr{k+1,j+i-1}{2};
      if val<splitn
        splitn = val;
        left = tr{j,k}{3}; right = tr{k+1,j+i-1}{3};
        pp = pum;
      end    
      
    end
    
    tree = [];
%    tree = struct;
    if isnumeric(left)
      tree.left = left; tree.lefttree = [];
    else
      tree.lefttree = left; tree.left = [left.left,left.right];
    end
    if isnumeric(right)
      tree.right = right; tree.righttree = [];
    else
      tree.righttree = right; tree.right = [right.left,right.right];
    end
    
    tr{j,j+i-1} = {splitn,pp,tree};
    
  
  end
end

tr = tr{1,end}{3};


return


function [T,tree_ind] = add_subtree(T,tree_ind,parent,letters,probs,opt)
switch length(letters)
 case 0
  error('cannot generate subtree of zero elements');
 case 1 
  % tree is a leaf
  T(tree_ind) = struct('parent', parent,'children',[],'leaves',letters);
 otherwise
  % general case
  T(tree_ind) = struct('parent', parent,'children',[],'leaves',letters);
  this_ind = tree_ind;
  [partition, part_probs] = get_partitions(letters, probs, opt);
  for i = 1:length(partition)
    T(this_ind).children(i) = tree_ind+1;
    [T,tree_ind] = add_subtree(T,tree_ind+1,this_ind,partition{i},part_probs{i},opt);
  end
end
tree_ind = tree_ind;
return

function [partition, part_probs] = get_partitions(letters,probs,opt)
probs = probs/sum(probs);
if(length(letters)<=opt.targets-sign(opt.alwaysreverse))
  % fewer boxes available - put one letter in each box.
  for i = 1:length(letters)
    partition{i} = letters(i);
    part_probs{i} = probs(i);
  end
else
  % sufficient boxes available
  ind(1) = 0;
  for i = 1:(opt.targets-sign(opt.alwaysreverse)-1)
      [dum,ind(i+1)] = min(abs(cumsum(probs)-i/(opt.targets-sign(opt.alwaysreverse))));
      % this prevents boxes from being empty:
      ind(i+1) = max(ind(i)+1,min(length(letters)-(opt.targets-sign(opt.alwaysreverse))+i,ind(i+1)));
      partition{i} = letters((ind(i)+1):ind(i+1));
      part_probs{i} = probs((ind(i)+1):ind(i+1));
  end
  partition{opt.targets-sign(opt.alwaysreverse)} = letters((ind(end)+1):end);
  part_probs{opt.targets-sign(opt.alwaysreverse)} = probs((ind(end)+1):end);
end
return