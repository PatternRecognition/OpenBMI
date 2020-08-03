classdef counter
    % Iteration command line visualization
    properties
        currIt = 1;
        N_newline = 25
        printChar = '.'
    end
    methods
      function obj = oneup(obj)
        fprintf(obj.printChar)
        if mod(obj.currIt,obj.N_newline)==0
          fprintf(' || %i\n', obj.currIt)
        end
        obj.currIt = obj.currIt+1;
      end
      function obj = reset(obj)
        obj.currIt = 1;
      end
    end
    
end