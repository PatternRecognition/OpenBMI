function epo = prepare_block_validation(epo,flag);
%PREPARE_BLOCK_VALIDATION prepares sample Divisions regarding block structure regarding epo.task for augcog
%
% usage:
%    epo = prepare_block_validation(epo,<flag=1>);
%   
% input:
%    epo    a usual epo structure of augcog
%    flag   0: each block is once test set
%           1: blocks with equal latin numbers are test set once
%
% output:
%    epo    with additionally fields divTr, divTe for xvalidation
%    epo    and block_marker as logical array
%
% Guido Dornhege, 27/04/2004

if ~exist('flag','var') | isempty(flag)
  flag = 1;
end

switch flag
 case 0
  epo.divTr = cell(1,length(epo.taskname));  
  epo.divTe = cell(1,length(epo.taskname));
  epo.block_marker = zeros(length(epo.taskname),size(epo.y,2));
  for i = 1:length(epo.taskname)
    epo.divTe{i} = {find(epo.task(i,:))};
    epo.divTr{i} = {setdiff(1:size(epo.y,2),epo.divTe{i}{1})};
    epo.block_marker(i,epo.divTe{i}{1})=1;
  end
 case 1
  taskn = epo.taskname;
  for i = 1:length(epo.className)
    ind = strmatch(epo.className{i},taskn);
    for j = 1:length(ind)
      taskn{ind(j)} = taskn{ind(j)}(length(epo.className{i})+2:end);
    end
  end
  epo.divTr = {};epo.divTe={};
  rem = 1:length(taskn);
  epo.block_marker = [];
  while ~isempty(rem)
    ta = taskn{rem(1)};
    ind = find(strcmp(ta,taskn));
    bla = find(sum(epo.task(ind,:),1));
    epo.divTe = {epo.divTe{:},{bla}};
    epo.divTr = {epo.divTr{:},{setdiff(1:size(epo.y,2),epo.divTe{end}{1})}};
    epo.block_marker = [epo.block_marker;zeros(1,size(epo.y,2))];
    epo.block_marker(end,bla)=1;
    rem = setdiff(rem,ind);
  end
end

  
  