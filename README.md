# needleman-wunsch
CS025 Final Project - Parallel Needleman-wunsch algorithm by Open_CL

nw.cl includes the first three implementations:
1) no-block parallel
2) block-wise parallel, serial within block
3) block-wise parallel, parallel within block
Modify the variable needleman in nw.py to choose which implementation to execute.
Modify string1 and string2 to determine what sequence the edit distance run on.
set show_progress = True to print out intermediate DP table.
set printresult = True to print out the result DP table of both parallel version and serial version.
This program will run the serial version once in the end to compare the time consumed and verify the correctness of the parallel edit distance by comparing the whole two DP tables.

To run this program, simply type
>python nw.py


nw_co.cl includes the forth implementation:
4) coalesced block read/write
Modify string1 and string2 to determine what sequence the edit distance run on.
set show_progress = True to print out intermediate DP table.
set printresult = True to print out the result DP table of serial version. The parallel version has only partial table stored because we only keep the parts that will be needed in the other iteration.
This program will run the serial version once in the end to compare the time consumed and verify the correctness of the parallel edit distance by comparing the last row and column of serial's DP tables and the last rows of parallel's row column table.

To run this program, simply type
>python nw_co.py
