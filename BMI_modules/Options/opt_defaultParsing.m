function out = opt_defaultParsing(def_opt, opt)
    def_field = fieldnames(def_opt);
    comp_field = fieldnames(opt);
    
    out = opt;

    diff_field = setdiff(def_field, comp_field);
    for i = 1:length(diff_field)
        out.(diff_field{i}) = def_opt.(diff_field{i});
    end
end