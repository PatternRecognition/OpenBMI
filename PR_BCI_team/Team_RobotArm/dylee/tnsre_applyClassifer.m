function [R, fv] = tnsre_applyClassifier(epo, Out, csp_w)

%     epo_proc = proc_filtButter(epo, 3, bandpass_filter);
%     epo_proc = proc_selectChannels(epo_proc, sub_channel); % Channel Selection
%     epo_proc = proc_commonAverageReference(epo_proc); % Applying Common Average Reference (CAR)  

    fv = proc_linearDerivation(epo, csp_w);
    fv = proc_variance(fv); 
    fv = proc_logarithm(fv);
    
%     fv.x = fv.x';
    
    R = applyClassifier(fv, 'wr_multiClass', Out.C);
    R = out2label(R);
    
    i = 0;
    for i=1:size(R, 2)
        switch R(i)
            case 1
                R(i) = 0; 
            case 2
                R(i) = 1;
       
        end        
    end    
end

