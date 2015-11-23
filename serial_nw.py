# Edit distance = 1 for insertion, deletion, and replacement
import numpy as np
import string
def edit_distance(strA, strB):
    lenA, lenB = len(strA), len(strB)
    if strA and strB:
        # DP table
        # DP[i,j] = edit distance between strA[:(i+1)] and strB[:(j+1)]
        dp = np.zeros((lenA+1, lenB+1), dtype=np.uint)
        # Initialize 
        dp[:,0] = np.arange(lenA+1, dtype=np.uint)
        dp[0,:] = np.arange(lenB+1, dtype=np.uint)
        # Update
        for i in range(1, lenA+1):
            for j in range(1, lenB+1):
                dp[i,j] = min(dp[i-1,j]+1,
                               dp[i,j-1]+1,
                               dp[i-1,j-1] + (0 if strA[i-1] == strB[j-1] else 1))
        return dp[lenA, lenB]
    else:
        return max(lenA, lenB)

if __name__ == '__main__':
    strA = string.ascii_uppercase * 100
    strB = list(strA)
    for i in (1,13,17,500,890,1500,2120):
        strB[i], strB[i+1] = strB[i+1], strB[i]
    strB = ''.join(strB)
    print edit_distance(strA, strB)