function H= plotCSPanalysis(fv, mnt, W, A, la, varargin)
%PLOTCSPANALYSIS - Show CSP filters and projections as topographies
%
%Synopsis:
% H= plotCSPanalysis(FV, MNT, CSP_W, CSP_A, CSP_EIG, <OPT>)
%
%Arguments:
% FV: data structure
% MNT: electrode montage, see getElectrodePositions
% CSP_W: CSP 'demixing' matrix
% CSP_A: CSP 'mixing' matrix
% CSP_EIG: eigenvalues of CSP patterns
% OPT: struct or property/value list of optional properties:
%  .row_layout: default 0.
%  .mark_patterns: vector of indices: these patterns are marked
%  .mark_style: the outline of the marked scalps is marked by setting
%      its properties to this property/value list (given in a cell array)
%  .colorOrder: can be used to give scalp outlines class specific colors.
%  .nComps: select top nComps patterns for both class (in two classes situation only)
%
%Returns:
% H: struct of handles to graphic objects

% Author(s): Benjamin Blankertz

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'colAx','sym', ...
                  'scalePos','none', ...
                  'title', 1, ...
                  'mark_patterns', [], ...
                  'mark_style', {'LineWidth',3}, ...
                  'row_layout', 0, ...
                  'axes_layout', {[0 0.02 0], [0.01 0.03 0.01]}, ...
                  'colorOrder', [], ...
                  'csp_clab', [], ...
                  'nComps', []);

if isequal(opt.title,1),
  if isfield(fv, 'title'),
    opt.title= fv.title;
  else
    opt.title= '';
  end
end

if ~exist('la','var'),
  la= [];
end

nClasses= size(fv.y,1);

if ~isempty(opt.nComps) & nClasses == 2
  d=size(W,2);
  W=W(:,[1:opt.nComps,d:-1:d-opt.nComps+1]);
  A=A([1:opt.nComps,d:-1:d-opt.nComps+1],:);
  la=la([1:opt.nComps,d:-1:d-opt.nComps+1]);
end

nPat= size(W,2);
nComps= ceil(nPat/nClasses);

if ischar(opt.mark_patterns),
  if ~strcmpi(opt.mark_patterns, 'all'),
    warning('unknown value for opt.mark_patterns ignored');
  end
  opt.mark_patterns= 1:nPat;
end

if isempty(opt.csp_clab),
  opt.csp_clab= cellstr([repmat('csp', [nPat 1]) int2str((1:nPat)')])';
end
if ~isfield(fv, 'origClab'),
  mnt= mnt_adaptMontage(mnt, fv.clab);
else
  mnt= mnt_adaptMontage(mnt, fv.origClab);
end

clf;
k= 0;
for rr= 1:nClasses,
  for cc= 1:nComps,
    k= k+1;
    if k>nPat,
      continue;
    end
    if opt.row_layout,
      ri= (rr-1)*nComps + cc;
      H.ax_filt(cc,rr)= ...
          subplotxl(2, nPat, [1 ri], opt.axes_layout{:});
      H.ax_pat(cc,rr)= ...
          subplotxl(2, nPat, [2 ri], opt.axes_layout{:});
    else
      H.ax_filt(cc,rr)= ...
          subplotxl(nComps, 2*nClasses, [cc rr*2-1], opt.axes_layout{:});
      H.ax_pat(cc,rr)= ...
          subplotxl(nComps, 2*nClasses, [cc rr*2], opt.axes_layout{:});
    end
    if ~isempty(A),
      H.scalp(cc,rr)= scalpPlot(mnt, A(k,:), opt);
    else
      H.scalp(cc,rr).head= [];
      H.scalp(cc,rr).nose= [];
    end
    axes(H.ax_filt(cc,rr));
    H.scalp_filt(cc,rr)= scalpPlot(mnt, W(:,k), opt);
    hh= [H.scalp(cc,rr).head, H.scalp(cc,rr).nose, ...
         H.scalp_filt(cc,rr).head, H.scalp_filt(cc,rr).nose];
    if ~isempty(opt.colorOrder),
      set(hh, 'Color',opt.colorOrder(rr,:));
    end
    if ismember(k, opt.mark_patterns),
      set(hh, opt.mark_style{:});
    end
    if isempty(la),
      label_str= opt.csp_clab{k};
    else
      label_str= sprintf('{\\bf %s}  [%.2f]', opt.csp_clab{k}, la(k));
    end
    if opt.row_layout,
      H.label(cc,rr)= title(label_str);
      set(H.label(cc,rr), 'FontSize',12);
    else
      axes(H.ax_pat(cc,rr));
      H.label(cc,rr)= ylabel(label_str);
    end
  end
end
set(H.label, 'Visible','on');

if ~isempty(opt.title) & ~isequal(opt.title,0),
  H.title= addTitle(untex(opt.title), 1, 1);
end
