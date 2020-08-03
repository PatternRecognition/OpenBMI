function fv= makeLetterTrials(fv)
%fv= makeLetterTrials(fv)

nTrialsPerBlock= 12;

[T, nChans, nTrials]= size(fv.x);
lett= unique(fv.base);
nLett= length(lett);
nRep= nTrials/(nTrialsPerBlock*nLett);

fv.x= reshape(fv.x, [T, nChans, nTrialsPerBlock, nRep, nLett]);
fv.x= permute(fv.x, [1 4 3 2 5]);
fv.x= reshape(fv.x, [T*nRep*nTrialsPerBlock, nChans, nLett]);


lett_matrix= reshape(['A':'Z', '1':'9', ' '], 6, 6)'; 
nHighPerLett= nRep*nTrialsPerBlock;
deviants= fv.y(1,:);
fv.y= zeros(36, nLett);
fv.lett= char(zeros(1, nLett));
fv.target= zeros(1, nLett);
for il= 1:nLett,
  iv= [1:12] + (il-1)*nHighPerLett;
  ir= find(deviants(iv));
  fv.target(il)= (sort(fv.code(iv(ir))) - [1 7]) * [6;1] + 1;
  fv.lett(il)= lett_matrix(fv.target(il));
  fv.y(fv.target(il),il)= 1;
end
fv.className= lett_matrix(:);
fv.nRep= nRep;

fv= rmfield(fv, {'t', 'base', 'code'});
