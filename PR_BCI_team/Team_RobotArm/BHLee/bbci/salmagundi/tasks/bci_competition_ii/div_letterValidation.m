function [divTr, divTe]= letterValidation(base, xTrials)
%% for albany P300

lett= unique(base);
nLett= length(lett);

[dTr, dTe]= sampleDivisions(1:nLett, xTrials);
divTr= dTr;
divTe= dTe;

for it= 1:length(divTr),
  for id= 1:length(divTr{1}),
    divTr{it}{id}= find(ismember(base, dTr{it}{id}));
    divTe{it}{id}= find(ismember(base, dTe{it}{id}));
  end
end
