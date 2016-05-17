function [ fv_test_mi ] = proc_test_fbcsp_mi( epo_test_mi, fb_csp_w, selectedIdx1, selectedIdx2, selectedIdx3, bnum )
        
        class1_V = []; class2_V=[]; class3_V=[];
        all_b =size(epo_test_mi.clab,2);
        b_clab = all_b / bnum;
        nComp_CSP = 24; % if you change the number of csp filters, you have to change it too. - hsan
        nClass = 3;    % 3
        epo_test_bp = epo_test_mi;
        for bp = 1:bnum
            epo_test_bp.clab = epo_test_mi.clab(:,(bp-1)*b_clab+1:bp*b_clab);
            epo_test_bp.x = epo_test_mi.x(:,(bp-1)*b_clab+1:bp*b_clab);
            fv_test_mi = proc_linearDerivation(epo_test_bp, fb_csp_w(:,(bp-1)*nComp_CSP+1:bp*nComp_CSP));
            fv_test_mi = proc_variance(fv_test_mi); fv_test_mi = proc_logarithm(fv_test_mi); 

            nCsp = size(fv_test_mi.x,2);
            elem = nCsp / nClass;  
            
            for c = 1:nClass
                if c == 1
                    class1_V = [class1_V fv_test_mi.x(:,(c-1)*elem+1:c*elem)];   % 1 8 1
                elseif c == 2
                        class2_V = [class2_V fv_test_mi.x(:,(c-1)*elem+1:c*elem)];   % 1 8 1
                else
                        class3_V = [class3_V fv_test_mi.x(:,(c-1)*elem+1:c*elem)];   % 1 8 1
                end
            end
        end

        %% feature selection
        % step 1: initialization
%         ext1_V = squeeze(class1_V);
%         ext2_V = squeeze(class2_V);
%         ext3_V = squeeze(class3_V);
        
        last = length(selectedIdx1);
        for i = 1:last
            selectedF1(1,i,:) = class1_V(1,selectedIdx1(i),:);   % set of selected features
        end
        last = length(selectedIdx2);
        for i = 1:last
            selectedF2(1,i,:) = class2_V(1,selectedIdx2(i),:);   % set of selected features
        end
        last = length(selectedIdx3);
        for i = 1:last
            selectedF3(1,i,:) = class3_V(1,selectedIdx3(i),:);   % set of selected features
        end
        
        fin_selectedF = [selectedF1, selectedF2, selectedF3];
        fv_test_mi.x = fin_selectedF;
        
        fv_test_mi.clab = [];
        nNewChans1 = size(selectedIdx1,2);
        fv_test_mi.clab= [fv_test_mi.clab cellstr([repmat('csp_LH',nNewChans1,1) int2str((1:nNewChans1)')])'];   % new csp clab
        nNewChans2 = size(selectedIdx2,2);
        fv_test_mi.clab= [fv_test_mi.clab cellstr([repmat('csp_RH',nNewChans2,1) int2str((1:nNewChans2)')])'];   % new csp clab
        nNewChans3 = size(selectedIdx3,2);
        fv_test_mi.clab= [fv_test_mi.clab cellstr([repmat('csp_F',nNewChans3,1) int2str((1:nNewChans3)')])'];   % new csp clab
end