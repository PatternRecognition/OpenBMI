function [ fv_train_act, old_csp_w, new_csp_w, uniqidx ] = proc_train_fbcsp( epo_train_act, bnum, normidx, flag_norm )
        
        fb_csp_w = [];
        dash_V = [];
        all_b =size(epo_train_act.clab,2);
        b_clab = all_b / bnum;
        epo_train_bp = epo_train_act;
        for bp = 1:bnum
            epo_train_bp.x = epo_train_act.x(:,(bp-1)*b_clab+1:bp*b_clab,:);
            [fv_train_act, csp_w] = proc_csp3(epo_train_bp, 4);

            dash_V = [dash_V fv_train_act.x];   % 1 72 137
            fb_csp_w = [fb_csp_w csp_w]; % 31 72
        end
        fv_train_act.x = dash_V;
        nNewChans = length(fb_csp_w);
        fv_train_act.clab= cellstr([repmat('csp',nNewChans,1) int2str((1:nNewChans)')])';   % new csp clab
        fv_train_act = proc_variance(fv_train_act); fv_train_act = proc_logarithm(fv_train_act);    % csp feature vectors
        fv_train_act.x(~isfinite(fv_train_act.x)) = 0;

        %% feature selection
        % step 1: initialization
        extV = squeeze(fv_train_act.x); % 72 137
      
        if flag_norm
            uniqidx = normidx.rest;
        else
            % compute the mutual information
            Iw = MI(extV', epo_train_act.y');
            [b, idx] = sort(Iw, 'descend');
            k = 4;  % the number of feature selection
            idx1 = idx(1:k);
            len_csp_w = size(fb_csp_w,2) / bnum;

            for d = 1:k
                m = fix(idx1(d)./len_csp_w);
                r = rem(idx1(d),len_csp_w);
                if r ~= 0
                    idx2(d) = ((m+1) .* len_csp_w + 1) - r;
                else
                    idx2(d) = (m-1) .* len_csp_w + 1;
                end
            end

            idxmix=[idx1, idx2];
            uniqidx = unique(idxmix);
          
        end        

        last = length(uniqidx);
        for j = 1:last
            new_csp_w(:,j) = fb_csp_w(:,uniqidx(j));
            selectedF(1,j,:) = fv_train_act.x(1,uniqidx(j),:);   % set of selected features
            selectedClab(1,j) = fv_train_act.clab(1,uniqidx(j));
        end
        fv_train_act.clab = selectedClab;
        fv_train_act.x = selectedF;
        old_csp_w = fb_csp_w;
end