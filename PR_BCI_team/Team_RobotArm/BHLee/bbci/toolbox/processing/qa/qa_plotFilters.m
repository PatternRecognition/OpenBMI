function H = qa_plotFilters(Cw, mnt, cfy_ivals)
%
% USAGE:     H = qa_plotFilters(Cw, mnt, cfy_ivals)
%
% Plot linear filters (e.g. LDA filters) as scalpmaps
%
% Simon Scholler, 
%

H = figure;
N = size(Cw,1);
for f = 1:N
    subplot(1,N,f)
    scalpPlot(mnt,Cw(:,f),'extrapolate',1)
    if nargin>2
        title([vec2str(cfy_ivals(f,:),'%d','-') ' ms'])
    end
end
suptitle('LDA filter(s)')
colorbars('delete')