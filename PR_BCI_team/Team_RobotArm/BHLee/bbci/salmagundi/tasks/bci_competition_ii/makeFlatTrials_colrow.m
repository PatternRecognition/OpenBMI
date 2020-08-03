function fv= makeFlatTrials_colrow(fv_col, fv_row)
%fv= makeFlatTrials_colrow(fv_col, fv_row)

nTrialsPerColrow= 6;
lett_matrix= reshape(['A':'Z', '1':'9', ' '], 6, 6)'; 

[nF_col, nTrials]= size(fv_col.x);
[nF_row, nTrials]= size(fv_row.x);

lett= unique(fv_col.base);
nLett= length(lett);
nRep= nTrials/(nTrialsPerColrow*nLett);

fv_col.x= reshape(fv_col.x, [nF_col, nTrialsPerColrow, nRep, nLett]);
fv_row.x= reshape(fv_row.x, [nF_row, nTrialsPerColrow, nRep, nLett]);
fv.x= cat(1, fv_col.x, fv_row.x);
sz= size(fv.x);
fv.x= reshape(fv.x, prod(sz(1:end-1)), nLett);
fv.className= lett_matrix(:);
fv.nRep= nRep;
fv.nF_col= nF_col;
fv.nF_row= nF_row;
fv.nC_col= 1;
fv.nC_row= 1;

if ~isfield(fv_col, 'y'),
  return;
end


nHighPerLett= nRep*nTrialsPerColrow;
fv.y= zeros(36, nLett);
fv.lett= char(zeros(1, nLett));
fv.target= zeros(1, nLett);
for il= 1:nLett,
  iv= [1:nTrialsPerColrow] + (il-1)*nHighPerLett;
  ir= find(fv_row.y(1,iv));
  ic= find(fv_col.y(1,iv));
  fv.target(il)= ([fv_col.code(iv(ic)) fv_row.code(iv(ir))] - [1 7]) ...
                 * [6;1] + 1;
  fv.lett(il)= lett_matrix(fv.target(il));
  fv.y(fv.target(il),il)= 1;
end
