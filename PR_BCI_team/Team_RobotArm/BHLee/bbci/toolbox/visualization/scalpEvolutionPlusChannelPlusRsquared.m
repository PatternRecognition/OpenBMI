function h = scalpEvolutionPlusChannelPlusRsquared(erp,epo_r,mnt,clab,ival,varargin)
% Display evolution of scalp topographies with additional r-squared plots.
% 
%
%Usage:
% H= scalpEvolutionPlusChannelPlusRsquared(ERP,EPO_R MNT, CLAB, IVAL, <OPTS>)
%
% Uses fig2subplot to add an additional set of r-squared scalp maps to the 
% output of scalpEvolutionPlusChannel.
%
% IN: erp  - struct of epoched EEG data. For convenience used classwise
%            averaged data, e.g., the result of proc_average.
%     epo_r -struct of epoched r-squared data
%     mnt  - struct defining an electrode montage
%     clab - label of the channel(s) which are to be displayed in the
%            ERP plot.
%     ival - [nIvals x 2]-sized array of interval, which are marked in the
%            ERP plot and for which scalp topographies are drawn.
%            When all interval are consequtive, ival can also be a
%            vector of interval borders.
%     opts - 
%            .erp_clim: color limits for the erp plots
%            .r_squared_clim: color limits for the r_squared plots
%
%      the opts struct is passed to scalpEvolutionPlusChannel
%
% OUT h - struct of handles to the created graphic objects.
%

% Author(s): Thomas Rost Oct 2010

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'erp_clim',[], ...
                 'r_squared_clim',[]);

% figure for scalpEvolutionPlusChannel
f_e = figure;
% enforce color limits
if length(opt.erp_clim) ==2,
opt.colAx = opt.erp_clim;
end
% produce the upper figure
h_e = scalpEvolutionPlusChannel(erp,mnt,clab,ival,opt);

% same for epo_r
f_r = figure;

if length(opt.r_squared_clim) ==2,
opt.colAx = opt.r_squared_clim;
end
h_r = scalpEvolutionPlusChannel(epo_r,mnt,[],ival,opt);

% get the scalp axis in the r_squared plot for rescaling
h_scalp = h_r.scalp(1).ax;
% get the position of the scalp axis
p_scalp = get(h_scalp,'position');
% get position of color bar
p_cb = get(h_r.cb,'position');
% set the height and bottom of cb to that of scalp
p_cb(2) = 0.5*(1.-0.75*p_scalp(4));
p_cb(4) = 0.75*p_scalp(4);
% now put the cb-position back in
set(h_r.cb,'position',p_cb);

% r_squared occupies the lower quarter of the new subplot
position = [0,0.25,1,0.75;0,0,1,0.24];

% make a new figure and return the handles
h =fig2subplot([f_e,f_r],'positions',position,'deleteFigs',1);


