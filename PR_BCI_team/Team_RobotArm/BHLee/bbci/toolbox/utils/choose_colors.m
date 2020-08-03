function cols= choose_colors(classes, colDef)
%cols= choose_colors(classes, colDef)
%
%Examples
%1 )epo is structure with
%   epo.className= {'left click', 'right no-click'}
%   colDef=  {'left*','right*','foot'; 
%             [1 0 0], [0 0.7 0], [0 0 1]};
%   choose_colors(epo.className, colDef);
%
%2) [??What is this needed for??: provide better example]
%.  classes= {'left*', '*no-click''};
%   colDef=  {'left click','right no-click','foot';
%             [1 0 0], [0 0.7 0], [0 0 1]};
%  choose_colors(classes, colDef);

nClasses= length(classes);
cols= zeros(nClasses, size(colDef{2,1},2));

for ic= 1:nClasses,
  cc= patpatternmatch(classes{ic}, colDef(1,:));
  cols(ic,:)= colDef{2,cc};
end
