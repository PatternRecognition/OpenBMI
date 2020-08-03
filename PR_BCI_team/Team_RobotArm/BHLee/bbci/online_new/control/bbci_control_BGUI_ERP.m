function [packet, state]= bbci_control_BGUI_ERP(cfy_out, state, event, opt)
%same as BBCI_CONTROL_ERP_SPELLER - just need another packet.
% 02-2012 Javier Pascual

    if(strcmp(opt.kind, 'MI')),
        packet= {['i:' opt.cfy], cfy_out};
    else,

        if isempty(state),
          state.counter= zeros(1, opt.nClasses);
          state.output= zeros(1, opt.nClasses);
        end

        this_cue= 1 + mod(event.desc-11, 10);
        state.counter(this_cue)= state.counter(this_cue) + 1;
        state.output(this_cue)= state.output(this_cue) + cfy_out;

        %fprintf('cfy_out = %.2f\n', cfy_out);
        %s = sprintf('output %d(%d) = %.2f %.2f %.2f %.2f %.2f %.2f\n',sum(state.counter),opt.nClasses*opt.nSequences,state.output(1),state.output(2),state.output(3),state.output(4),state.output(5),state.output(6) );
        %fprintf(s);
        %msgbox(s);

        if sum(state.counter) >= opt.nClasses*opt.nSequences,
          idx= find(state.counter>0);  % avoid divide by zero
          state.output(idx)= state.output(idx) ./ state.counter(idx);
          [max_score, selected_class]= min(state.output);

          %fprintf('  -> selected class = %d\n',selected_class );

          packet= {'i:erp_output', selected_class};
          state.counter(:)= 0;
          state.output(:)= 0;
        else
          packet= [];
        end;
    end;    
end
