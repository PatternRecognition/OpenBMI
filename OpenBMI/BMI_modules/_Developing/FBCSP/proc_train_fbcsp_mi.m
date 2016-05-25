function [ fv_train_mi, old_csp_w, fin_csp_w, uniqidx1, uniqidx2, uniqidx3] = proc_train_fbcsp_mi( epo_train_mi, bnum, normidx, flag_norm )

        cl_csp_w1 =[];  cl_csp_w2 =[];  cl_csp_w3 =[];
        class1_V = []; class2_V=[]; class3_V=[];
        fb_csp_w = [];
        fin_csp_w = [];
        all_b =size(epo_train_mi.clab,2);
        b_clab = all_b / bnum;
        nClass = size(epo_train_mi.y,1);    % 3
        epo_train_bp = epo_train_mi;
        for bp = 1:bnum
            epo_train_bp.clab = epo_train_mi.clab(:,(bp-1)*b_clab+1:bp*b_clab);
            epo_train_bp.x = epo_train_mi.x(:,(bp-1)*b_clab+1:bp*b_clab,:);
            [fv_train_mi, csp_w]=  proc_multicsp_hsan(epo_train_bp, 4);
            fv_train_mi = proc_variance(fv_train_mi); fv_train_mi= proc_logarithm(fv_train_mi);
            fv_train_mi.x(~isfinite(fv_train_mi.x)) = 0;

            [s nCsp totTrials] = size(fv_train_mi.x);
            elem = nCsp / nClass;  
            
            for c = 1:nClass
                if c == 1
                    class1_V = [class1_V fv_train_mi.x(:,(c-1)*elem+1:c*elem,:)];   % 1 8 69
                elseif c == 2
                    class2_V = [class2_V fv_train_mi.x(:,(c-1)*elem+1:c*elem,:)];   % 1 8 69
                else
                    class3_V = [class3_V fv_train_mi.x(:,(c-1)*elem+1:c*elem,:)];   % 1 8 69
                end
            end
            
            fb_csp_w = [ fb_csp_w csp_w ];
            
            cl_csp_w1 = [cl_csp_w1 fb_csp_w(:,1:elem)];  % 31x8*9
            cl_csp_w2 = [cl_csp_w2 fb_csp_w(:,elem+1:2*elem)];  % 31x8*9
            cl_csp_w3 = [cl_csp_w3 fb_csp_w(:,2*elem+1:end)];  % 31x8*9
            
        end

        %% feature selection
        % step 1: initialization
        ext1_V = squeeze(class1_V); % 8*9 69
        ext2_V = squeeze(class2_V);
        ext3_V = squeeze(class3_V);
 
        if flag_norm
            uniqidx1 = normidx.left;
            uniqidx2 = normidx.right;
            uniqidx3 = normidx.foot;
        else
            ylabels1(1,:) = epo_train_mi.y(1,:);
            ylabels1(2,:) = 0;
            ylabels2(1,:) = epo_train_mi.y(2,:);
            ylabels2(2,:) = 0;
            ylabels3(1,:) = epo_train_mi.y(3,:);
            ylabels3(2,:) = 0;
            for yi = 1:length(epo_train_mi.y)
                for yc = 1:size(epo_train_mi.y,1)
                    if yc == 1 && epo_train_mi.y(yc,yi) ~= 1
                        ylabels1(2,yi) = 1;
                    elseif yc == 2 && epo_train_mi.y(yc,yi) ~= 1
                            ylabels2(2,yi) = 1;
                    elseif yc == 3 && epo_train_mi.y(yc,yi) ~= 1
                                ylabels3(2,yi) = 1;
                    end
                end
            end

            % compute the mutual information
            Iw1 = MI(ext1_V', ylabels1');
            Iw2 = MI(ext2_V', ylabels2');
            Iw3 = MI(ext3_V', ylabels3');

            [b1, idx1] = sort(Iw1, 'descend');
            [b2, idx2] = sort(Iw2, 'descend');
            [b3, idx3] = sort(Iw3, 'descend');

            k = 4;  % the number of feature selection
            idx1 = idx1(1:k);
            idx2 = idx2(1:k);
            idx3 = idx3(1:k);
            ind_len = size(cl_csp_w1,2) ./ bnum;

            for d = 1:k
                m1 = fix(idx1(d)./ind_len);
                r1 = rem(idx1(d),ind_len);
                m2 = fix(idx2(d)./ind_len);
                r2 = rem(idx2(d),ind_len);
                m3 = fix(idx3(d)./ind_len);
                r3 = rem(idx3(d),ind_len);

                if r1 ~= 0
                    nidx1(d) = ((m1+1) .* ind_len + 1) - r1;
                else
                    nidx1(d) = ((m1-1) .* ind_len) + 1;
                end

                if r2 ~= 0
                    nidx2(d) = ((m2+1) .* ind_len + 1) - r2;
                else
                    nidx2(d) = ((m2-1) .* ind_len) + 1;
                end

                if r3 ~= 0
                    nidx3(d) = ((m3+1) .* ind_len + 1) - r3;
                else
                    nidx3(d) = ((m3-1) .* ind_len) + 1;
                end
            end

            idxmix1 = [idx1, nidx1];
            idxmix2 = [idx2, nidx2];
            idxmix3 = [idx3, nidx3];

            uniqidx1 = unique(idxmix1);
            uniqidx2 = unique(idxmix2);
            uniqidx3 = unique(idxmix3);
        end

        last = length(uniqidx1);
        for j = 1:last
            new_csp_w1(:,j) = cl_csp_w1(:,uniqidx1(j));
            selectedF1(1,j,:) = class1_V(1,uniqidx1(j),:);   % set of selected features
        end

        last = length(uniqidx2);
        for j = 1:last
            new_csp_w2(:,j) = cl_csp_w2(:,uniqidx2(j));
            selectedF2(1,j,:) = class2_V(1,uniqidx2(j),:);   % set of selected features
        end

        last = length(uniqidx3);
        for j = 1:last
            new_csp_w3(:,j) = cl_csp_w3(:,uniqidx3(j));
            selectedF3(1,j,:) = class3_V(1,uniqidx3(j),:);   % set of selected features
        end
        
        fin_csp_w = [new_csp_w1, new_csp_w2, new_csp_w3];
        fin_selectedF = [selectedF1, selectedF2, selectedF3];
        
        old_csp_w = fb_csp_w;
        fv_train_mi.x = fin_selectedF;
        
        fv_train_mi.clab = [];
        nNewChans1 = size(new_csp_w1,2);
        fv_train_mi.clab= [fv_train_mi.clab cellstr([repmat('csp_LH',nNewChans1,1) int2str((1:nNewChans1)')])'];   % new csp clab
        nNewChans2 = size(new_csp_w2,2);
        fv_train_mi.clab= [fv_train_mi.clab cellstr([repmat('csp_RH',nNewChans2,1) int2str((1:nNewChans2)')])'];   % new csp clab
        nNewChans3 = size(new_csp_w3,2);
        fv_train_mi.clab= [fv_train_mi.clab cellstr([repmat('csp_F',nNewChans3,1) int2str((1:nNewChans3)')])'];   % new csp clab
        
end