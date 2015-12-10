CS025 Final Project - Parallel Needleman-Wunsch
===============================================

nw.cl includes the first three parallel implementations:

1. no-block parallel
2. block-wise parallel, serial within block
3. block-wise parallel, parallel within block

* Modify the variable needleman in nw.py to choose which implementation to execute.
* Set string1 and string2 to specify string inputs.
* Set show_progress = True to print out intermediate DP table.
* Set printresult = True to print out the result DP table of both parallel and serial versions.

This program will run the serial version once in the end to compare the time consumed and verify the correctness of the parallel edit distance by comparing the two DP tables.

To run this program, simply run

>python nw.py


nw_co.cl includes the fourth parallel implementation:

4. coalesced block read/write

* Set string1 and string2 to specify string inputs.
* Set show_progress = True to print out intermediate DP table.
* Set printresult = True to print out the result DP table of serial version. The parallel version has only partial table stored because we only keep the parts that will be needed in the next iterations.

This program will run the serial version once in the end to compare the time consumed and verify the correctness of the parallel edit distance by comparing the last row and column of the serial DP tables and the last rows of the parallel implementation's row column tables.

To run this program, simply run

>python nw_co.py
