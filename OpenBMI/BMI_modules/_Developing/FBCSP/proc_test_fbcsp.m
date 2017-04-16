function [ fv_test_act ] = proc_test_fbcsp( epo_test_act, fb_csp_w, selectedIdx, bnum )
        % fb_csp_w : old csp w values
        
        dash_V = [];
        all_b =size(epo_test_act.clab,2);
        b_clab = all_b / bnum;
        nComp_CSP = 8;  % if you change the number of csp filters, you have to change it too. - hsan
        epo_test_bp = epo_test_act;
        for bp = 1:bnum
            epo_test_bp.x = epo_test_act.x(:,(bp-1)*b_clab+1:bp*b_clab,:);
            fv_test_act = proc_linearDerivation(epo_test_bp, fb_csp_w(:,(bp-1)*nComp_CSP+1:bp*nComp_CSP));
         
            dash_V = [dash_V fv_test_act.x];   % 1 72 137
        end
        fv_test_act.x = dash_V;
        nNewChans = length(fb_csp_w);
        fv_test_act.clab= cellstr([repmat('csp',nNewChans,1) int2str((1:nNewChans)')])';   % new csp clab
        fv_test_act = proc_variance(fv_test_act); fv_test_act = proc_logarithm(fv_test_act);

        %% feature selection
        % step 1: initialization
%         extV = squeeze(dash_V); % 72 137
        
        last = length(selectedIdx);
        for i = 1:last
            selectedF(1,i,:) = fv_test_act.x(1,selectedIdx(i),:);   % set of selected features
            selectedClab(1,i) = fv_test_act.clab(1,selectedIdx(i));
        end
        fv_test_act.clab = selectedClab;
        fv_test_act.x = selectedF;
end