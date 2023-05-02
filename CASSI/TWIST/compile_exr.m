%============================================================================
%
% Codename: Bird Scanner (Yale Computer Graphics Group)
%
% Copyright (C) 2011-12 Min H. Kim. All rights reserved.
%
% GNU General Public License Usage
% Alternatively, this file may be used under the terms of the GNU General
% Public License version 3.0 as published by the Free Software Foundation
% and appearing in the file LICENSE.GPL included in the packaging of this
% file. Please review the following information to ensure the GNU General
% Public License version 3.0 requirements will be met:
% http://www.gnu.org/copyleft/gpl.html.
%
%============================================================================

%--------------------------------------------------------------
% for Mac Lion/Snow Leopard (64-bit)
%--------------------------------------------------------------
% kimo
mex exrinfo.cpp  -lIlmImf -lIex -lImath -lHalf -I/usr/local/include/OpenEXR -L/usr/local/lib
%mex exrread.cpp -lIlmImf -lIex -lImath -lHalf -I/usr/local/include/OpenEXR -L/usr/local/lib
%mex exrwrite.cpp -lIlmImf -lIex -lImath -lHalf -I/usr/local/include/OpenEXR -L/usr/local/lib
% edgar
%mex exrinfo.cpp util.cpp ImfToMatlab.cpp -lIlmImf -lIex -lImath -lHalf -I/usr/local/include/OpenEXR -L/usr/local/lib
mex exrread.cpp util.cpp -lIlmImf -lIex -lImath -lHalf -I/usr/local/include/OpenEXR -L/usr/local/lib
mex exrwrite.cpp util.cpp -lIlmImf -lIex -lImath -lHalf -I/usr/local/include/OpenEXR -L/usr/local/lib
mex exrreadchannels.cpp util.cpp -lIlmImf -lIex -lImath -lHalf -I/usr/local/include/OpenEXR -L/usr/local/lib
mex exrwritechannels.cpp util.cpp MatlabToImf.cpp -lIlmImf -lIex -lImath -lHalf -I/usr/local/include/OpenEXR -L/usr/local/lib

%--------------------------------------------------------------
% for Windows7 (64-bit)
%--------------------------------------------------------------

% 32 bits
mex exrinfo.cpp  util.cpp -lIlmImf -lIex -lImath -lHalf -Ic:/mkcassi/exr/include -Lc:/mkcassi/exr/lib/Win32/Release
mex exrread.cpp util.cpp -lzlibwapi -lIlmImf -lIlmThread -lIex -lImath -lHalf -Ic:/mkcassi/exr/include -Lc:/mkcassi/exr/lib/Win32/Release
mex exrwrite.cpp util.cpp -lzlibwapi  -lIlmImf -lIex -lImath -lIlmThread -lHalf -Ic:/mkcassi/exr/include -Lc:/mkcassi/exr/lib/Win32/Release
mex exrreadchannels.cpp util.cpp -lzlibwapi -lIlmImf -lIlmThread -lIex -lImath -lHalf -Ic:/mkcassi/exr/include -Lc:/mkcassi/exr/lib/Win32/Release
mex exrwritechannels.cpp MatlabToImf.cpp util.cpp -lzlibwapi -lIlmImf -lIlmThread -lIex -lImath -lHalf -Ic:/mkcassi/exr/include -Lc:/mkcassi/exr/lib/Win32/Release

% 64 bits
mex exrinfo.cpp  util.cpp -lIlmImf -lIex -lImath -lHalf -Ic:/mkcassi/exr/include -Lc:/mkcassi/exr/lib/x64/Release
mex exrread.cpp util.cpp -lzlibwapi -lIlmImf -lIlmThread -lIex -lImath -lHalf -Ic:/mkcassi/exr/include -Lc:/mkcassi/exr/lib/x64/Release
mex exrwrite.cpp util.cpp -lzlibwapi  -lIlmImf -lIex -lImath -lIlmThread -lHalf -Ic:/mkcassi/exr/include -Lc:/mkcassi/exr/lib/x64/Release
mex exrreadchannels.cpp util.cpp -lzlibwapi -lIlmImf -lIlmThread -lIex -lImath -lHalf -Ic:/mkcassi/exr/include -Lc:/mkcassi/exr/lib/x64/Release
mex exrwritechannels.cpp MatlabToImf.cpp util.cpp -lzlibwapi -lIlmImf -lIlmThread -lIex -lImath -lHalf -Ic:/mkcassi/exr/include -Lc:/mkcassi/exr/lib/x64/Release