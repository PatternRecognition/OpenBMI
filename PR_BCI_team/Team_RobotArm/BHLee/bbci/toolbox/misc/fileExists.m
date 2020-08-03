function bool = fileExists(fileName)
% 
% FILEEXISTS Check if file exists.
% fileExists(filename)
% 
% Return true if the file exists.
% 
        import java.io.*;
        a=File(fileName);
        bool=a.exists();
end
