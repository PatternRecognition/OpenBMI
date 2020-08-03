function [divTr, divTe]= letterOneOut(base)
%% for albany P300

lett= unique(base);
nLett= length(lett);
divTr= {cell(1, nLett)};
divTe= {cell(1, nLett)};

for il= 1:nLett,
  divTr{1}{il}= find(base~=lett(il));
  divTe{1}{il}= find(base==lett(il));
end
