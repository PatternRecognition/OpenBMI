function [repIdx, eqcl]= xval_choose_repIdx(fv, check_consistency)
% helper function for xvalidation and select_model.

if ~exist('check_consistency','var'),
  check_consistency= 1;
end

if isfield(fv, 'bidx'),
  %% check equivalence classes as defined by field bidx for consistency
  %% and choose a representative for each.
  valid= find(any(fv.y~=0 & ~isnan(fv.y),1));
  eqcl= unique(fv.bidx(valid));
  repIdx= zeros(1, length(eqcl));
  for kk= 1:length(eqcl),
    %% all members that belong to the kk-th equivalence class
    mm= find(fv.bidx==eqcl(kk));
    ca= find(sum(fv.y(:,mm),2)>0);
    if check_consistency & length(ca)>1,
      error(sprintf('inconsistent labels in equivalence class with bidx %d',...
                    eqcl(kk)));
    end
    repIdx(kk)= mm(1);
  end
else
  repIdx= find(any(fv.y,1));
  eqcl = repIdx;
end
