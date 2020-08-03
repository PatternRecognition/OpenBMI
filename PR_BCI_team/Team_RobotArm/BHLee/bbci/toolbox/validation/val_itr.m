function varargout = val_itr(cm, time, varargin)

% doc to come
%
% Martijn, 2011

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'mode', 'wolpaw'); %{'wolpaw','schloegl', 'all'}
                  

switch ndims(cm),
    case 1,
        error('Confusion matrix should at least have 2 dimensions.');
        
    case 3,
        if length(time) == 1,
            warning('Assuming equal time for all received CM.');
            time = repmat(time, 1, size(cm,3));
        elseif length(time) ~= size(cm, 3),
            error('Time should have same length as the number of CM''s');
        end
end

keep = [];
for cmId = 1:size(cm,3),
    curCM = squeeze(cm(:,:,cmId));
    N = size(curCM,1);
    
    if ismember(opt.mode, {'wolpaw', 'all'}),
        resIdx = 1;
        P = sum(curCM(find(eye(N))))/sum(sum(curCM));
        itr(resIdx,cmId) = bitrate(P,N);
        if cmId ==1, keep = 1; end;
    end
       
    if ismember(opt.mode, {'schloegl', 'all'}),
        resIdx = 2;
        curCM(~curCM) = .0000001; % prevent NaN
        px = sum(curCM,2)/sum(sum(curCM));   
        pyx = curCM ./ repmat(sum(curCM,2), 1,N);
        pyx(~pyx) = .0000001; % prevent NaN due to perfect class
        py = sum(repmat(px', N,1) .* pyx,1);            
        itr(resIdx,cmId) = sum(px' * (pyx.*log2(pyx))) - sum(py.*log2(py));
        if cmId ==1, keep = [keep 2]; end;
    end  
end

itr(itr <0) = 0;
itr = itr(keep,:);
varargout{1} = itr;
if ~isempty(time),
    varargout{2} = itr./(repmat(time, size(itr,1),1)/60);
end
end
